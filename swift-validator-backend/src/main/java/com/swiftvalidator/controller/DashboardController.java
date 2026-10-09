package com.swiftvalidator.controller;

import com.swiftvalidator.dto.DashboardStatsDTO;
import com.swiftvalidator.service.AnalysisService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/dashboard")
@RequiredArgsConstructor
public class DashboardController {

    private final AnalysisService analysisService;

    @GetMapping("/stats")
    public ResponseEntity<DashboardStatsDTO> stats(@AuthenticationPrincipal UserDetails userDetails) {
        return ResponseEntity.ok(analysisService.getDashboardStats(userDetails.getUsername()));
    }
}
