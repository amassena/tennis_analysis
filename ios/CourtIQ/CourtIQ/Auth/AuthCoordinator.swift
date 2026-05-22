import Foundation
import AuthenticationServices
import Combine

/// Owns the sign-in lifecycle and publishes the current auth state for the UI.
///
/// State machine:
///   .unknown    — launch state, before we've checked the keychain.
///   .signedOut  — no JWT (or it was rejected by the server). Show AuthGateView.
///   .signedIn   — we have a JWT and have confirmed it via /api/me.
///
/// Apple identity tokens are short-lived and we only need them once; we trade
/// them for our own 30-day JWT and use that for all subsequent calls.
@MainActor
final class AuthCoordinator: NSObject, ObservableObject {
    enum AuthState: Equatable {
        case unknown
        case signedOut
        case signedIn(userHash: String)
    }

    @Published private(set) var state: AuthState = .unknown
    @Published private(set) var lastError: String?
    @Published private(set) var isSigningIn = false

    private var signInContinuation: CheckedContinuation<Void, Error>?

    /// Call once on app launch. If we have a stored JWT, hit /api/me to
    /// confirm it still works; otherwise drop to .signedOut.
    func bootstrap() async {
        if TokenStore.load() == nil {
            state = .signedOut
            return
        }
        do {
            let me: MeResponse = try await APIClient.get(path: "api/me")
            state = .signedIn(userHash: me.user_hash)
        } catch APIClient.APIError.unauthorized {
            TokenStore.clear()
            state = .signedOut
        } catch {
            // Network error — keep .unknown so we don't bounce the user to
            // sign-in on a flaky connection. Retried by user action.
            lastError = error.localizedDescription
        }
    }

    /// Kicked off by AuthGateView when the user taps Sign in with Apple.
    /// Receives the completion result from `SignInWithAppleButton`.
    func handleSignInCompletion(_ result: Result<ASAuthorization, Error>) {
        Task {
            isSigningIn = true
            defer { isSigningIn = false }
            switch result {
            case .failure(let error):
                let cancelled = (error as? ASAuthorizationError)?.code == .canceled
                lastError = cancelled ? nil : error.localizedDescription
            case .success(let auth):
                await exchangeAppleCredential(auth)
            }
        }
    }

    func signOut() {
        TokenStore.clear()
        state = .signedOut
        lastError = nil
    }

    // MARK: - Private

    private func exchangeAppleCredential(_ auth: ASAuthorization) async {
        guard let credential = auth.credential as? ASAuthorizationAppleIDCredential else {
            lastError = "Unexpected Apple credential type"
            return
        }
        guard
            let tokenData = credential.identityToken,
            let identityToken = String(data: tokenData, encoding: .utf8)
        else {
            lastError = "Apple returned no identity token"
            return
        }

        do {
            let body = AuthAppleRequest(identity_token: identityToken)
            let response: AuthAppleResponse = try await APIClient.post(
                path: "api/auth/apple",
                body: body,
                requireAuth: false
            )
            try TokenStore.save(response.jwt)
            state = .signedIn(userHash: response.user_hash)
            lastError = nil
        } catch {
            lastError = error.localizedDescription
        }
    }
}

// MARK: - Wire types

private struct AuthAppleRequest: Encodable {
    let identity_token: String
}

private struct AuthAppleResponse: Decodable {
    let jwt: String
    let user_hash: String
    let gallery_url: String
    let expires_at: Int
}

private struct MeResponse: Decodable {
    let user_hash: String
    let video_count: Int
    let gallery_url: String
    let created_at: String
}
