package com.pathrank.ads.server.security;

import com.nimbusds.jose.jwk.source.ImmutableSecret;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.oauth2.jose.jws.MacAlgorithm;
import org.springframework.security.oauth2.jwt.JwtDecoder;
import org.springframework.security.oauth2.jwt.JwtEncoder;
import org.springframework.security.oauth2.jwt.JwtValidators;
import org.springframework.security.oauth2.jwt.NimbusJwtDecoder;
import org.springframework.security.oauth2.jwt.NimbusJwtEncoder;

/** HS256 signing and verification with one shared secret (ads.security.jwt-secret, base64, >= 32 bytes). */
@Configuration
public class JwtConfig {

    static SecretKey key(SecurityProperties p) {
        if (p.jwtSecret() == null || p.jwtSecret().isBlank()) {
            throw new IllegalStateException("ads.security.jwt-secret is required (base64, at least 32 bytes)");
        }
        byte[] raw;
        try {
            raw = Base64.getDecoder().decode(p.jwtSecret());
        } catch (IllegalArgumentException e) {
            raw = p.jwtSecret().getBytes(StandardCharsets.UTF_8);
        }
        if (raw.length < 32) throw new IllegalStateException("ads.security.jwt-secret must be at least 32 bytes for HS256");
        return new SecretKeySpec(raw, "HmacSHA256");
    }

    @Bean
    JwtEncoder jwtEncoder(SecurityProperties p) {
        return new NimbusJwtEncoder(new ImmutableSecret<>(key(p)));
    }

    @Bean
    JwtDecoder jwtDecoder(SecurityProperties p) {
        NimbusJwtDecoder d = NimbusJwtDecoder.withSecretKey(key(p)).macAlgorithm(MacAlgorithm.HS256).build();
        // signature + exp/nbf (default) + our issuer
        d.setJwtValidator(JwtValidators.createDefaultWithIssuer(p.issuer()));
        return d;
    }
}
