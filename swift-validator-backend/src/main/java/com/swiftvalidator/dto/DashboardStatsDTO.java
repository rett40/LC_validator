package com.swiftvalidator.dto;

import lombok.Data;
import lombok.Builder;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;

@Data @Builder @NoArgsConstructor @AllArgsConstructor
public class DashboardStatsDTO {
    private long totalAnalyses;
    private long criticalCount;
    private long highCount;
    private long mediumCount;
    private long cleanCount;
    private long last24hCount;
    private long paymentBlockedCount;
    private Double avgUcpScore;
}
