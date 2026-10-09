package com.swiftvalidator.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.List;

public class AdminDTO {

    /* ── Agent list ── */
    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class AgentResponse {
        private Long   id;
        private String username;
        private String email;
        private String role;
        private Boolean enabled;
        private LocalDateTime createdAt;
        private long   analysisCount;
    }

    /* ── Create agent ── */
    @Data
    public static class CreateAgentRequest {
        @NotBlank @Size(min = 3, max = 50) private String username;
        @NotBlank @Email                   private String email;
        @NotBlank @Size(min = 8)           private String password;
        private String role = "AGENT";
    }

    /* ── Update agent ── */
    @Data
    public static class UpdateAgentRequest {
        @Email private String email;
        private String role;
    }

    /* ── Toggle status ── */
    @Data
    public static class ToggleStatusRequest {
        private boolean enabled;
    }

    /* ── Admin dashboard stats ── */
    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class AdminStatsResponse {
        private long   totalAgents;
        private long   totalAnalyses;
        private long   todayAnalyses;
        private long   acceptCount;
        private long   reviewCount;
        private long   rejectCount;
        private double avgScore;
        private List<DayCount>        weeklyTrend;
        private List<DecisionSegment> decisionBreakdown;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class DayCount {
        private String day;
        private long   count;
    }

    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class DecisionSegment {
        private String label;
        private long   value;
        private String color;
    }

    /* ── Analysis history item (per agent) ── */
    @Data @Builder @NoArgsConstructor @AllArgsConstructor
    public static class AnalysisHistoryItem {
        private Long          id;
        private String        lcReference;
        private String        decision;
        private int           score;
        private LocalDateTime analyzedAt;
        private int           warnings;
        private int           contradictions;
    }
}
