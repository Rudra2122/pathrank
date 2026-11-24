package com.pathrank.ads.server.store;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.sql.SQLException;
import org.junit.jupiter.api.Test;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.orm.ObjectOptimisticLockingFailureException;
import org.springframework.transaction.TransactionSystemException;

/** Which failures mean "someone else wrote first, try again" and which are real errors. */
class JpaBudgetStoreRetryTest {
    @Test
    void retriesConcurrencyFailures() {
        assertTrue(JpaBudgetStore.isRetryable(new ObjectOptimisticLockingFailureException(AdvertiserBudget.class, "adv-1")));
        assertTrue(JpaBudgetStore.isRetryable(new DataIntegrityViolationException("duplicate key")));
        // CockroachDB SERIALIZABLE restart, wrapped the way it arrives from a commit
        assertTrue(JpaBudgetStore.isRetryable(new TransactionSystemException("commit failed",
                new RuntimeException(new SQLException("restart transaction", "40001")))));
    }

    @Test
    void doesNotRetryRealErrors() {
        assertFalse(JpaBudgetStore.isRetryable(new IllegalStateException("bug")));
        assertFalse(JpaBudgetStore.isRetryable(new RuntimeException(new SQLException("syntax error", "42601"))));
    }
}
