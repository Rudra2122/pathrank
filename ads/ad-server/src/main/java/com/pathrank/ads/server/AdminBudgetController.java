package com.pathrank.ads.server;

import com.pathrank.ads.core.BudgetStore;
import com.pathrank.ads.server.store.JpaBudgetStore;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Positive;
import jakarta.validation.constraints.PositiveOrZero;
import java.util.LinkedHashMap;
import java.util.Map;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Admin-only door straight into the budget store, used by ads/bench/overspend.py to hammer one
 * advertiser's budget from two ad server processes at once. It runs the exact {@code charge()} a
 * click runs, without needing an impression first.
 */
@RestController
@RequestMapping("/admin/budget")
public class AdminBudgetController {
    public record ChargeRequest(@NotBlank String advertiserId, @Positive long dailyBudgetMicros,
                                @PositiveOrZero long priceMicros) {}

    private final BudgetStore budget;

    public AdminBudgetController(BudgetStore budget) {
        this.budget = budget;
    }

    @PostMapping("/charge")
    public Map<String, Object> charge(@Valid @RequestBody ChargeRequest r) {
        boolean ok = budget.charge(r.advertiserId(), r.dailyBudgetMicros(), r.priceMicros());
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("charged", ok);
        return out;
    }

    @GetMapping("/{advertiserId}")
    public Map<String, Object> get(@PathVariable("advertiserId") String advertiserId) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("advertiserId", advertiserId);
        out.put("store", budget instanceof JpaBudgetStore ? "jpa" : "memory");
        out.put("spentMicros", budget instanceof JpaBudgetStore j ? j.spentFromDatabase(advertiserId) : budget.spent(advertiserId));
        return out;
    }
}
