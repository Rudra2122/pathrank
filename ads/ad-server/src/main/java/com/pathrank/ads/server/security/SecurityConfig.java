package com.pathrank.ads.server.security;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.convert.converter.Converter;
import org.springframework.http.HttpMethod;
import org.springframework.security.authentication.AbstractAuthenticationToken;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.oauth2.jwt.Jwt;
import org.springframework.security.oauth2.server.resource.authentication.JwtAuthenticationToken;
import org.springframework.security.web.SecurityFilterChain;

/**
 * Who may call what. Stateless: every request carries a Bearer JWT, no sessions, so CSRF doesn't
 * apply. No token or a bad/expired one is 401; a valid token with the wrong role is 403; an
 * advertiser asking for another advertiser's campaign is 403 from the controller.
 *
 * <p>ads.security.enabled=false turns all of it off (the original benchmarks, and the auth-cost
 * comparison in ads/bench/compare.sh).
 */
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Bean
    SecurityFilterChain filterChain(HttpSecurity http, @Value("${ads.security.enabled:true}") boolean enabled)
            throws Exception {
        http.csrf(AbstractHttpConfigurer::disable)
            .sessionManagement(s -> s.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
            .httpBasic(AbstractHttpConfigurer::disable)
            .formLogin(AbstractHttpConfigurer::disable);
        if (!enabled) {
            http.authorizeHttpRequests(a -> a.anyRequest().permitAll());
            return http.build();
        }
        http.authorizeHttpRequests(a -> a
                .requestMatchers("/actuator/health/**", "/actuator/info", "/actuator/prometheus").permitAll()
                .requestMatchers(HttpMethod.POST, "/auth/token").permitAll()
                // Spring Boot renders 400/404 bodies by forwarding to /error; without this, a 404 for a
                // missing campaign would come back as 403 from the denyAll below
                .requestMatchers("/error").permitAll()
                .requestMatchers(HttpMethod.POST, "/ads/auction", "/ads/click").hasAnyRole(Roles.SERVICE, Roles.ADMIN)
                .requestMatchers("/ads/stats", "/admin/**", "/actuator/**").hasRole(Roles.ADMIN)
                .requestMatchers("/api/**").hasAnyRole(Roles.ADVERTISER, Roles.ADMIN)
                .anyRequest().denyAll())
            .oauth2ResourceServer(o -> o.jwt(j -> j.jwtAuthenticationConverter(new RolesConverter())));
        return http.build();
    }

    /** "roles": ["ADVERTISER"] in the token becomes the ROLE_ADVERTISER authority. */
    static final class RolesConverter implements Converter<Jwt, AbstractAuthenticationToken> {
        @Override
        public AbstractAuthenticationToken convert(Jwt jwt) {
            Collection<GrantedAuthority> out = new ArrayList<>();
            List<String> roles = jwt.getClaimAsStringList(Roles.CLAIM_ROLES);
            if (roles != null) for (String r : roles) out.add(new SimpleGrantedAuthority("ROLE_" + r));
            return new JwtAuthenticationToken(jwt, out, jwt.getSubject());
        }
    }

}
