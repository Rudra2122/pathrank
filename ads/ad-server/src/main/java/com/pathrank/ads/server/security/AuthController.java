package com.pathrank.ads.server.security;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import java.time.Instant;
import java.util.List;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.oauth2.jose.jws.MacAlgorithm;
import org.springframework.security.oauth2.jwt.JwsHeader;
import org.springframework.security.oauth2.jwt.JwtClaimsSet;
import org.springframework.security.oauth2.jwt.JwtEncoder;
import org.springframework.security.oauth2.jwt.JwtEncoderParameters;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

/** POST /auth/token: username + password in, a signed JWT carrying the role (and advertiser id) out. */
@RestController
@ConditionalOnProperty(name = "ads.security.enabled", havingValue = "true", matchIfMissing = true)
public class AuthController {
    public record TokenRequest(@NotBlank String username, @NotBlank String password) {}

    public record TokenResponse(String accessToken, String tokenType, long expiresIn, String role, String advertiserId) {}

    private final UserDirectory users;
    private final JwtEncoder encoder;
    private final SecurityProperties props;

    public AuthController(UserDirectory users, JwtEncoder encoder, SecurityProperties props) {
        this.users = users;
        this.encoder = encoder;
        this.props = props;
    }

    @PostMapping("/auth/token")
    public ResponseEntity<TokenResponse> token(@Valid @RequestBody TokenRequest req) {
        return users.authenticate(req.username(), req.password())
                .map(a -> {
                    Instant now = Instant.now();
                    JwtClaimsSet.Builder claims = JwtClaimsSet.builder()
                            .issuer(props.issuer())
                            .subject(a.username())
                            .issuedAt(now)
                            .expiresAt(now.plusSeconds(props.tokenTtlSeconds()))
                            .claim(Roles.CLAIM_ROLES, List.of(a.role()));
                    if (a.advertiserId() != null) claims.claim(Roles.CLAIM_ADVERTISER, a.advertiserId());
                    String token = encoder.encode(JwtEncoderParameters.from(
                            JwsHeader.with(MacAlgorithm.HS256).build(), claims.build())).getTokenValue();
                    return ResponseEntity.ok(new TokenResponse(token, "Bearer", props.tokenTtlSeconds(), a.role(),
                            a.advertiserId()));
                })
                .orElseGet(() -> ResponseEntity.status(HttpStatus.UNAUTHORIZED).build());
    }
}
