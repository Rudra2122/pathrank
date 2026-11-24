package com.pathrank.ads.server.store;

import com.pathrank.ads.core.BudgetStore;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import java.sql.SQLException;
import java.time.Clock;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.dao.ConcurrencyFailureException;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;

/**
 * Daily advertiser spend in PostgreSQL or CockroachDB, so more than one ad server can charge clicks
 * for the same advertiser without overspending.
 *
 * <p>Charging: read the advertiser's row, check the budget, add the price, write it back with
 * {@code @Version} optimistic locking. If another server wrote the row in between, the write fails
 * and this server re-reads and tries again (up to {@code ads.budget.max-attempts}, with a little
 * random backoff). On CockroachDB, which runs every transaction SERIALIZABLE, a conflict can also
 * surface as a serialization failure (SQLSTATE 40001); that is retried the same way.
 *
 * <p>Auction filtering ({@link #canAfford}) never touches the database: it reads a local copy of
 * today's spend that is refreshed every {@code ads.ctr-refresh-ms}. That copy can be stale, so the
 * auction may occasionally pick an ad whose budget just ran out on another server; the charge is
 * still refused, it just costs one wasted slot. Correctness lives in {@link #charge}, speed in
 * {@link #canAfford}.
 */
@Component
@ConditionalOnProperty(name = "ads.budget-store", havingValue = "jpa")
public class JpaBudgetStore implements BudgetStore {
    private static final Logger log = LoggerFactory.getLogger(JpaBudgetStore.class);

    private final AdvertiserBudgetRepository repo;
    private final TransactionTemplate tx;
    private final Clock clock;
    private final int maxAttempts;
    private final Map<String, Long> spentCache = new ConcurrentHashMap<>();
    private final Counter retries, conflictsGivenUp, charged, refused;
    private final Timer chargeTimer;

    @Autowired
    public JpaBudgetStore(AdvertiserBudgetRepository repo, PlatformTransactionManager txm, MeterRegistry registry,
                          @Value("${ads.budget.max-attempts:30}") int maxAttempts) {
        this(repo, txm, registry, maxAttempts, Clock.systemUTC());
    }

    JpaBudgetStore(AdvertiserBudgetRepository repo, PlatformTransactionManager txm, MeterRegistry registry,
                   int maxAttempts, Clock clock) {
        this.repo = repo;
        this.tx = new TransactionTemplate(txm);
        this.clock = clock;
        this.maxAttempts = maxAttempts;
        this.retries = Counter.builder("ads.budget.retries").description("optimistic-lock / serialization retries")
                .register(registry);
        this.conflictsGivenUp = Counter.builder("ads.budget.conflicts_given_up").register(registry);
        this.charged = Counter.builder("ads.budget.charges").tag("result", "charged").register(registry);
        this.refused = Counter.builder("ads.budget.charges").tag("result", "over_budget").register(registry);
        this.chargeTimer = Timer.builder("ads.budget.charge.latency").publishPercentileHistogram().register(registry);
    }

    private LocalDate today() {
        return LocalDate.now(clock.withZone(ZoneOffset.UTC));
    }

    @Override
    public boolean canAfford(String advertiserId, long dailyBudgetMicros, long maxPriceMicros) {
        return spentCache.getOrDefault(advertiserId, 0L) + maxPriceMicros <= dailyBudgetMicros;
    }

    @Override
    public boolean charge(String advertiserId, long dailyBudgetMicros, long priceMicros) {
        if (priceMicros < 0) throw new IllegalArgumentException("price must be >= 0");
        return chargeTimer.record(() -> chargeWithRetry(advertiserId, dailyBudgetMicros, priceMicros));
    }

    private boolean chargeWithRetry(String advertiserId, long dailyBudgetMicros, long priceMicros) {
        for (int attempt = 1; ; attempt++) {
            try {
                long[] spentAfter = new long[1];
                Boolean ok = tx.execute(status -> {
                    LocalDate day = today();
                    AdvertiserBudget b = repo.findById(advertiserId).orElseGet(() -> new AdvertiserBudget(advertiserId, day));
                    b.rollTo(day);
                    if (b.getSpentMicros() + priceMicros > dailyBudgetMicros) {
                        spentAfter[0] = b.getSpentMicros();
                        return false;
                    }
                    b.add(priceMicros);
                    repo.saveAndFlush(b);   // flush inside the transaction so a version conflict is thrown here
                    spentAfter[0] = b.getSpentMicros();
                    return true;
                });
                spentCache.merge(advertiserId, spentAfter[0], Math::max);
                if (Boolean.TRUE.equals(ok)) charged.increment(); else refused.increment();
                return Boolean.TRUE.equals(ok);
            } catch (RuntimeException e) {
                if (!isRetryable(e)) throw e;
                retries.increment();
                if (attempt >= maxAttempts) {
                    conflictsGivenUp.increment();
                    log.warn("giving up charging {} after {} attempts: {}", advertiserId, attempt, e.toString());
                    return false;   // fail closed: never charge without a committed write
                }
                backoff(attempt);
            }
        }
    }

    /**
     * Lost an optimistic-lock race, lost an insert race on a brand-new advertiser row, or (CockroachDB)
     * got a SERIALIZABLE restart. All three mean "someone else wrote first, re-read and try again".
     */
    static boolean isRetryable(Throwable e) {
        for (Throwable t = e; t != null; t = t.getCause()) {
            if (t instanceof ConcurrencyFailureException || t instanceof DataIntegrityViolationException) return true;
            if (t instanceof jakarta.persistence.OptimisticLockException) return true;
            if (t instanceof SQLException s && "40001".equals(s.getSQLState())) return true;
            if (t.getCause() == t) break;
        }
        return false;
    }

    private static void backoff(int attempt) {
        long capMs = Math.min(50, 1L << Math.min(attempt, 6));
        try {
            Thread.sleep(ThreadLocalRandom.current().nextLong(0, capMs + 1));
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        }
    }

    /** What the database says right now (used by the console and the overspend benchmark). */
    public long spentFromDatabase(String advertiserId) {
        LocalDate day = today();
        return repo.findById(advertiserId).filter(b -> day.equals(b.getBudgetDay())).map(AdvertiserBudget::getSpentMicros)
                .orElse(0L);
    }

    @Override
    public long spent(String advertiserId) {
        return spentCache.getOrDefault(advertiserId, 0L);
    }

    /** Pull today's spend for every advertiser so {@link #canAfford} sees what other servers charged. */
    @Scheduled(fixedDelayString = "${ads.ctr-refresh-ms:500}")
    public void refresh() {
        try {
            LocalDate day = today();
            Map<String, Long> fresh = new ConcurrentHashMap<>();
            for (AdvertiserBudget b : repo.findByBudgetDay(day)) fresh.put(b.getAdvertiserId(), b.getSpentMicros());
            spentCache.keySet().retainAll(fresh.keySet());
            spentCache.putAll(fresh);
        } catch (RuntimeException e) {
            log.warn("budget refresh failed, keeping previous values: {}", e.toString());
        }
    }

    @Override
    public void resetAll() {
        // rows roll over to the new day on their next charge (see AdvertiserBudget.rollTo), so every
        // instance can run this at midnight without racing the others on a delete
        spentCache.clear();
    }
}
