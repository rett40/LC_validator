package com.swiftvalidator.config;

import com.swiftvalidator.entity.User;
import com.swiftvalidator.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.CommandLineRunner;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
@Slf4j
public class DataInitializer implements CommandLineRunner {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;

    @Override
    public void run(String... args) {
        ensureAdmin("admin", "admin@swiftvalidator.local", "Admin@1234");
    }

    private void ensureAdmin(String username, String email, String password) {
        userRepository.findByUsername(username).ifPresentOrElse(
            existing -> {
                if (!"ADMIN".equals(existing.getRole())) {
                    existing.setRole("ADMIN");
                    userRepository.save(existing);
                    log.info("Updated user '{}' role to ADMIN", username);
                }
                // Reset password to ensure it matches the configured one
                existing.setPasswordHash(passwordEncoder.encode(password));
                existing.setEnabled(true);
                userRepository.save(existing);
                log.info("Admin user '{}' credentials refreshed", username);
            },
            () -> {
                User admin = User.builder()
                    .username(username)
                    .email(email)
                    .passwordHash(passwordEncoder.encode(password))
                    .role("ADMIN")
                    .enabled(true)
                    .build();
                userRepository.save(admin);
                log.info("Admin user '{}' created", username);
            }
        );
    }
}
