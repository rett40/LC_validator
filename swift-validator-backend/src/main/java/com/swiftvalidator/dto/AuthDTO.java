package com.swiftvalidator.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

public class AuthDTO {

    @Data
    public static class LoginRequest {
        @NotBlank private String username;
        @NotBlank private String password;
    }

    @Data
    public static class RegisterRequest {
        @NotBlank @Size(min = 3, max = 50) private String username;
        @NotBlank @Email private String email;
        @NotBlank @Size(min = 8) private String password;
    }

    @Data @lombok.Builder @lombok.NoArgsConstructor @lombok.AllArgsConstructor
    public static class AuthResponse {
        private String token;
        private String tokenType;
        private Long userId;
        private String username;
        private String role;
        private long expiresIn;
    }
}
