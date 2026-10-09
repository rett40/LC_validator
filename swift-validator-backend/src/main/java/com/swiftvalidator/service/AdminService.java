package com.swiftvalidator.service;

import com.swiftvalidator.dto.AdminDTO;
import com.swiftvalidator.entity.SwiftAnalysis;
import com.swiftvalidator.entity.User;
import com.swiftvalidator.repository.SwiftAnalysisRepository;
import com.swiftvalidator.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.PageRequest;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.TextStyle;
import java.util.*;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class AdminService {

    private final UserRepository         userRepo;
    private final SwiftAnalysisRepository analysisRepo;
    private final PasswordEncoder         passwordEncoder;

    // ── Decision mapping: DB riskLevel → ACCEPT / REVIEW / REJECT ──
    private static final List<String> ACCEPT_LEVELS = List.of("CLEAN", "LOW");
    private static final List<String> REVIEW_LEVELS = List.of("MEDIUM", "HIGH");
    private static final List<String> REJECT_LEVELS = List.of("CRITICAL");

    /* ──────────────────────── STATS ──────────────────────── */

    @Transactional(readOnly = true)
    public AdminDTO.AdminStatsResponse getStats() {
        long totalAgents   = userRepo.countActiveAgents();
        long totalAnalyses = analysisRepo.count();
        long todayAnalyses = analysisRepo.countSince(LocalDateTime.now().withHour(0).withMinute(0).withSecond(0).withNano(0));

        long acceptCount = analysisRepo.countByRiskLevelIn(ACCEPT_LEVELS);
        long reviewCount = analysisRepo.countByRiskLevelIn(REVIEW_LEVELS);
        long rejectCount = analysisRepo.countByRiskLevelIn(REJECT_LEVELS);

        Double avg = analysisRepo.avgUcpScoreAll();
        double avgScore = avg != null ? Math.round(avg * 10.0) / 10.0 : 0.0;

        List<AdminDTO.DayCount> weeklyTrend = buildWeeklyTrend();

        List<AdminDTO.DecisionSegment> breakdown = List.of(
            AdminDTO.DecisionSegment.builder().label("ACCEPTÉ").value(acceptCount).color("#22c55e").build(),
            AdminDTO.DecisionSegment.builder().label("RÉVISION").value(reviewCount).color("#f59e0b").build(),
            AdminDTO.DecisionSegment.builder().label("REJETÉ").value(rejectCount).color("#ef4444").build()
        );

        return AdminDTO.AdminStatsResponse.builder()
            .totalAgents(totalAgents)
            .totalAnalyses(totalAnalyses)
            .todayAnalyses(todayAnalyses)
            .acceptCount(acceptCount)
            .reviewCount(reviewCount)
            .rejectCount(rejectCount)
            .avgScore(avgScore)
            .weeklyTrend(weeklyTrend)
            .decisionBreakdown(breakdown)
            .build();
    }

    private List<AdminDTO.DayCount> buildWeeklyTrend() {
        LocalDateTime since = LocalDateTime.now().minusDays(6).withHour(0).withMinute(0).withSecond(0);
        List<Object[]> rows = analysisRepo.countByDaySince(since);

        // Build a map date → count from DB results
        // Hibernate 6 + PostgreSQL may return LocalDate or java.sql.Date depending on driver version
        Map<LocalDate, Long> dbMap = new LinkedHashMap<>();
        for (Object[] row : rows) {
            if (row[0] == null) continue;
            LocalDate d;
            if (row[0] instanceof LocalDate) {
                d = (LocalDate) row[0];
            } else if (row[0] instanceof java.sql.Date) {
                d = ((java.sql.Date) row[0]).toLocalDate();
            } else {
                d = LocalDate.parse(row[0].toString());
            }
            long cnt = ((Number) row[1]).longValue();
            dbMap.put(d, cnt);
        }

        // Fill all 7 days (including days with 0 analyses)
        List<AdminDTO.DayCount> result = new ArrayList<>();
        for (int i = 6; i >= 0; i--) {
            LocalDate day = LocalDate.now().minusDays(i);
            String label  = day.getDayOfWeek().getDisplayName(TextStyle.SHORT, Locale.FRENCH);
            long count    = dbMap.getOrDefault(day, 0L);
            result.add(AdminDTO.DayCount.builder().day(capitalize(label)).count(count).build());
        }
        return result;
    }

    /* ──────────────────────── AGENTS ──────────────────────── */

    @Transactional(readOnly = true)
    public List<AdminDTO.AgentResponse> getAgents() {
        return userRepo.findByRoleNot("ADMIN").stream()
            .map(u -> {
                long count = analysisRepo.countByUserId(u.getId());
                return AdminDTO.AgentResponse.builder()
                    .id(u.getId())
                    .username(u.getUsername())
                    .email(u.getEmail())
                    .role(u.getRole())
                    .enabled(u.getEnabled())
                    .createdAt(u.getCreatedAt())
                    .analysisCount(count)
                    .build();
            })
            .collect(Collectors.toList());
    }

    @Transactional
    public AdminDTO.AgentResponse createAgent(AdminDTO.CreateAgentRequest req) {
        if (userRepo.existsByUsername(req.getUsername())) {
            throw new IllegalArgumentException("Username already taken: " + req.getUsername());
        }
        if (userRepo.existsByEmail(req.getEmail())) {
            throw new IllegalArgumentException("Email already registered: " + req.getEmail());
        }

        String role = (req.getRole() != null && req.getRole().equals("ADMIN")) ? "ADMIN" : "AGENT";

        User user = User.builder()
            .username(req.getUsername())
            .email(req.getEmail())
            .passwordHash(passwordEncoder.encode(req.getPassword()))
            .role(role)
            .enabled(true)
            .build();

        User saved = userRepo.save(user);
        log.info("Admin created new agent: {}", saved.getUsername());

        return AdminDTO.AgentResponse.builder()
            .id(saved.getId())
            .username(saved.getUsername())
            .email(saved.getEmail())
            .role(saved.getRole())
            .enabled(saved.getEnabled())
            .createdAt(saved.getCreatedAt())
            .analysisCount(0)
            .build();
    }

    @Transactional
    public AdminDTO.AgentResponse updateAgent(Long id, AdminDTO.UpdateAgentRequest req) {
        User user = userRepo.findById(id)
            .orElseThrow(() -> new RuntimeException("Agent not found: " + id));

        if (req.getEmail() != null && !req.getEmail().equals(user.getEmail())) {
            if (userRepo.existsByEmail(req.getEmail())) {
                throw new IllegalArgumentException("Email already registered: " + req.getEmail());
            }
            user.setEmail(req.getEmail());
        }
        if (req.getRole() != null) {
            user.setRole(req.getRole());
        }

        User saved = userRepo.save(user);
        long count = analysisRepo.countByUserId(saved.getId());
        return toAgentResponse(saved, count);
    }

    @Transactional
    public AdminDTO.AgentResponse toggleStatus(Long id, boolean enabled) {
        User user = userRepo.findById(id)
            .orElseThrow(() -> new RuntimeException("Agent not found: " + id));
        user.setEnabled(enabled);
        User saved = userRepo.save(user);
        long count = analysisRepo.countByUserId(saved.getId());
        log.info("Agent {} status → {}", saved.getUsername(), enabled);
        return toAgentResponse(saved, count);
    }

    @Transactional
    public void deleteAgent(Long id) {
        User user = userRepo.findById(id)
            .orElseThrow(() -> new RuntimeException("Agent not found: " + id));
        if ("ADMIN".equals(user.getRole())) {
            throw new IllegalStateException("Cannot delete an ADMIN account");
        }
        userRepo.delete(user);
        log.info("Admin deleted agent id={}", id);
    }

    /* ──────────────────────── HISTORY ──────────────────────── */

    @Transactional(readOnly = true)
    public List<AdminDTO.AnalysisHistoryItem> getAgentHistory(Long agentId) {
        userRepo.findById(agentId)
            .orElseThrow(() -> new RuntimeException("Agent not found: " + agentId));

        List<SwiftAnalysis> analyses = analysisRepo.findByUserIdOrdered(agentId, PageRequest.of(0, 50));

        return analyses.stream().map(a -> AdminDTO.AnalysisHistoryItem.builder()
            .id(a.getId())
            .lcReference(a.getLcNumber())
            .decision(mapDecision(a.getRiskLevel()))
            .score(a.getUcpComplianceScore() != null ? a.getUcpComplianceScore() : 0)
            .analyzedAt(a.getCreatedAt())
            .warnings(a.getNUcpWarnings() != null ? a.getNUcpWarnings() : 0)
            .contradictions(a.getNCriticalClauses() != null ? a.getNCriticalClauses() : 0)
            .build()
        ).collect(Collectors.toList());
    }

    /* ── helpers ── */

    private String mapDecision(String riskLevel) {
        if (riskLevel == null) return "REVIEW";
        return switch (riskLevel.toUpperCase()) {
            case "CLEAN", "LOW"  -> "ACCEPT";
            case "CRITICAL"      -> "REJECT";
            default              -> "REVIEW";
        };
    }

    private AdminDTO.AgentResponse toAgentResponse(User u, long count) {
        return AdminDTO.AgentResponse.builder()
            .id(u.getId()).username(u.getUsername()).email(u.getEmail())
            .role(u.getRole()).enabled(u.getEnabled()).createdAt(u.getCreatedAt())
            .analysisCount(count).build();
    }

    private String capitalize(String s) {
        if (s == null || s.isEmpty()) return s;
        return Character.toUpperCase(s.charAt(0)) + s.substring(1);
    }
}
