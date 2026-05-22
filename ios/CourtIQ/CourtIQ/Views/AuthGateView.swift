import SwiftUI
import AuthenticationServices

/// Full-screen welcome + Sign in with Apple gate.
///
/// Shown when `AuthCoordinator.state == .signedOut`.
struct AuthGateView: View {
    @EnvironmentObject var auth: AuthCoordinator

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 28) {
                Spacer()

                VStack(spacing: 14) {
                    Image(systemName: "figure.tennis")
                        .font(.system(size: 72, weight: .light))
                        .foregroundColor(.white)
                    Text("Tennis Uploader")
                        .font(.system(size: 30, weight: .bold))
                        .foregroundColor(.white)
                    Text("Record or pick a video; we'll process it and add it to your gallery.")
                        .font(.system(size: 15))
                        .foregroundColor(.white.opacity(0.7))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 36)
                }

                Spacer()

                VStack(spacing: 12) {
                    SignInWithAppleButton(.signIn) { request in
                        request.requestedScopes = [.email]
                    } onCompletion: { result in
                        auth.handleSignInCompletion(result)
                    }
                    .signInWithAppleButtonStyle(.white)
                    .frame(height: 50)
                    .cornerRadius(8)
                    .padding(.horizontal, 28)
                    .disabled(auth.isSigningIn)
                    .opacity(auth.isSigningIn ? 0.6 : 1)

                    if auth.isSigningIn {
                        ProgressView()
                            .tint(.white)
                    }

                    if let err = auth.lastError, !err.isEmpty {
                        Text(err)
                            .font(.footnote)
                            .foregroundColor(.red)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 32)
                    }
                }
                .padding(.bottom, 44)
            }
        }
    }
}
