package com.swiftvalidator.controller;

import com.swiftvalidator.dto.AdminDTO;
import com.swiftvalidator.service.AdminService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/v1/admin")
@RequiredArgsConstructor
@PreAuthorize("hasRole('ADMIN')")
public class AdminController {

    private final AdminService adminService;

    /* ── Dashboard stats ── */
    @GetMapping("/stats")
    public ResponseEntity<AdminDTO.AdminStatsResponse> getStats() {
        return ResponseEntity.ok(adminService.getStats());
    }

    /* ── Agent list ── */
    @GetMapping("/users")
    public ResponseEntity<List<AdminDTO.AgentResponse>> getAgents() {
        return ResponseEntity.ok(adminService.getAgents());
    }

    /* ── Create agent ── */
    @PostMapping("/users")
    public ResponseEntity<AdminDTO.AgentResponse> createAgent(
            @Valid @RequestBody AdminDTO.CreateAgentRequest req) {
        return ResponseEntity.status(HttpStatus.CREATED).body(adminService.createAgent(req));
    }

    /* ── Update agent (email, role) ── */
    @PutMapping("/users/{id}")
    public ResponseEntity<AdminDTO.AgentResponse> updateAgent(
            @PathVariable Long id,
            @RequestBody AdminDTO.UpdateAgentRequest req) {
        return ResponseEntity.ok(adminService.updateAgent(id, req));
    }

    /* ── Toggle enabled/disabled ── */
    @PatchMapping("/users/{id}/status")
    public ResponseEntity<AdminDTO.AgentResponse> toggleStatus(
            @PathVariable Long id,
            @RequestBody AdminDTO.ToggleStatusRequest req) {
        return ResponseEntity.ok(adminService.toggleStatus(id, req.isEnabled()));
    }

    /* ── Delete agent ── */
    @DeleteMapping("/users/{id}")
    public ResponseEntity<Void> deleteAgent(@PathVariable Long id) {
        adminService.deleteAgent(id);
        return ResponseEntity.noContent().build();
    }

    /* ── History per agent ── */
    @GetMapping("/users/{id}/analyses")
    public ResponseEntity<List<AdminDTO.AnalysisHistoryItem>> getAgentHistory(
            @PathVariable Long id) {
        return ResponseEntity.ok(adminService.getAgentHistory(id));
    }
}
