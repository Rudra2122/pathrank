package com.pathrank.ads.server.security;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

/** The configured users, with passwords BCrypt-hashed at startup so plaintext never sits in memory after boot. */
@Component
public class UserDirectory {
    public record Account(String username, String passwordHash, String role, String advertiserId) {}

    private final PasswordEncoder encoder = new BCryptPasswordEncoder();
    private final Map<String, Account> byName = new HashMap<>();
    // compared against when the username is unknown, so a wrong username takes as long as a wrong password
    private final String dummyHash = encoder.encode("not-a-real-password");

    public UserDirectory(SecurityProperties p) {
        for (SecurityProperties.User u : p.usersOrEmpty()) {
            String role = u.role() == null ? "" : u.role().toUpperCase();
            if (!role.equals(Roles.ADMIN) && !role.equals(Roles.SERVICE) && !role.equals(Roles.ADVERTISER)) {
                throw new IllegalStateException("user " + u.username() + ": unknown role " + u.role());
            }
            if (role.equals(Roles.ADVERTISER) && (u.advertiserId() == null || u.advertiserId().isBlank())) {
                throw new IllegalStateException("advertiser user " + u.username() + " needs an advertiserId");
            }
            byName.put(u.username(), new Account(u.username(), encoder.encode(u.password()), role, u.advertiserId()));
        }
    }

    public Optional<Account> authenticate(String username, String password) {
        Account a = username == null ? null : byName.get(username);
        if (a == null) {
            encoder.matches(password == null ? "" : password, dummyHash);
            return Optional.empty();
        }
        return encoder.matches(password == null ? "" : password, a.passwordHash()) ? Optional.of(a) : Optional.empty();
    }
}
