package com.swiftvalidator.repository;

import com.swiftvalidator.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.List;
import java.util.Optional;

public interface UserRepository extends JpaRepository<User, Long> {
    Optional<User> findByUsername(String username);
    Optional<User> findByEmail(String email);
    boolean existsByUsername(String username);
    boolean existsByEmail(String email);

    List<User> findByRoleNot(String role);

    @Query("SELECT COUNT(u) FROM User u WHERE u.role != 'ADMIN' AND u.enabled = true")
    long countActiveAgents();
}
