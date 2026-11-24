package com.pathrank.ads.server.store;

import java.time.LocalDate;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

public interface AdvertiserBudgetRepository extends JpaRepository<AdvertiserBudget, String> {
    List<AdvertiserBudget> findByBudgetDay(LocalDate day);
}
