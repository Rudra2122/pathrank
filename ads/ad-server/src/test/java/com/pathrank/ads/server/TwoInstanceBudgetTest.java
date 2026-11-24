package com.pathrank.ads.server;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.pathrank.ads.core.BudgetStore;
import com.pathrank.ads.server.store.JpaBudgetStore;
import io.micrometer.core.instrument.MeterRegistry;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ConfigurableApplicationContext;
import org.testcontainers.containers.CockroachContainer;
import org.testcontainers.containers.JdbcDatabaseContainer;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

/**
 * Two complete ad server instances (two Spring contexts, two connection pools, two local spend
 * caches) charging the same advertiser at the same time through JpaBudgetStore. With a budget of
 * exactly 1,000 clicks and 3,000 attempts, exactly 1,000 must be charged and the row must end at
 * exactly the budget, on PostgreSQL and on CockroachDB.
 *
 * <p>ads/bench/overspend.py does the same thing with two separate JVM processes over HTTP.
 */
@Testcontainers
class TwoInstanceBudgetTest {
    static final long BUDGET = 1_000_000;   // $1.00
    static final long PRICE = 1_000;        // $0.001 per click -> exactly 1,000 clicks fit
    static final int ATTEMPTS_PER_INSTANCE = 1_500;
    static final int THREADS_PER_INSTANCE = 8;

    @Container
    static final PostgreSQLContainer<?> POSTGRES = new PostgreSQLContainer<>("postgres:16-alpine");

    @Container
    static final CockroachContainer COCKROACH = new CockroachContainer("cockroachdb/cockroach:v24.2.4");

    @Test
    void postgresNeverOverspends() throws Exception {
        run(POSTGRES, "org.hibernate.dialect.PostgreSQLDialect", "pg");
    }

    @Test
    void cockroachNeverOverspends() throws Exception {
        run(COCKROACH, "org.hibernate.dialect.CockroachDialect", "crdb");
    }

    private ConfigurableApplicationContext instance(JdbcDatabaseContainer<?> db, String dialect) {
        return new SpringApplicationBuilder(AdServerApplication.class).run(
                "--server.port=0",
                "--ads.campaigns-path=classpath:campaigns-test.json",
                "--ads.budget-store=jpa",
                "--ads.security.enabled=false",
                "--spring.kafka.bootstrap-servers=localhost:1",        // never used: no auctions here
                "--spring.data.redis.port=1",
                "--ads.ctr-refresh-ms=600000",
                "--spring.datasource.url=" + db.getJdbcUrl(),
                "--spring.datasource.username=" + db.getUsername(),
                "--spring.datasource.password=" + db.getPassword(),
                "--spring.jpa.properties.hibernate.dialect=" + dialect);
    }

    private void run(JdbcDatabaseContainer<?> db, String dialect, String tag) throws Exception {
        String advertiser = "adv-race-" + tag;
        try (ConfigurableApplicationContext a = instance(db, dialect);
             ConfigurableApplicationContext b = instance(db, dialect)) {
            List<BudgetStore> stores = List.of(a.getBean(BudgetStore.class), b.getBean(BudgetStore.class));
            AtomicInteger charged = new AtomicInteger();
            ExecutorService pool = Executors.newFixedThreadPool(THREADS_PER_INSTANCE * 2);
            List<Future<?>> futures = new ArrayList<>();
            long t0 = System.nanoTime();
            for (BudgetStore s : stores) {
                for (int t = 0; t < THREADS_PER_INSTANCE; t++) {
                    futures.add(pool.submit(() -> {
                        for (int i = 0; i < ATTEMPTS_PER_INSTANCE / THREADS_PER_INSTANCE; i++) {
                            if (s.charge(advertiser, BUDGET, PRICE)) charged.incrementAndGet();
                        }
                    }));
                }
            }
            for (Future<?> f : futures) f.get(5, TimeUnit.MINUTES);
            pool.shutdown();
            double secs = (System.nanoTime() - t0) / 1e9;

            long spent = ((JpaBudgetStore) stores.get(0)).spentFromDatabase(advertiser);
            double retries = a.getBean(MeterRegistry.class).counter("ads.budget.retries").count()
                    + b.getBean(MeterRegistry.class).counter("ads.budget.retries").count();
            System.out.printf("[%s] 2 instances x %d threads: charged %d of %d attempts, spent %d/%d micros, "
                            + "%.0f retries, %.1f s%n", tag, THREADS_PER_INSTANCE,
                    charged.get(), ATTEMPTS_PER_INSTANCE * 2, spent, BUDGET, retries, secs);

            assertEquals(BUDGET / PRICE, charged.get(), "exactly the budget's worth of clicks is charged");
            assertEquals(BUDGET, spent, "the row ends at exactly the budget, never over");
        }
    }
}
