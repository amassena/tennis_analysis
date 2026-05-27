import SwiftUI

/// First-run welcome shown once per device after the user has signed in.
/// Doesn't request any permissions — camera + photo-library prompts are
/// JIT (fire only when the user actually taps Record or Pick). Push
/// notifications aren't used at all (status surfaces via the Recent
/// section on the Upload tab).
struct WelcomeSheet: View {
    @Binding var isPresented: Bool
    let userHash: String

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(spacing: 20) {
                    Image(systemName: "tennis.racket")
                        .font(.system(size: 64, weight: .light))
                        .foregroundColor(.accentColor)
                        .padding(.top, 40)

                    Text("Welcome to Tennis Uploader")
                        .font(.title.weight(.semibold))
                        .multilineTextAlignment(.center)

                    Text("Record or pick a tennis video. We'll detect every shot, render slow-motion clips, and stack your form against the pros.")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 20)

                    VStack(alignment: .leading, spacing: 18) {
                        Bullet(systemImage: "video.fill",
                               title: "Record or pick",
                               detail: "Tap the + button — record a session or pick a clip from Photos.")
                        Bullet(systemImage: "clock.arrow.circlepath",
                               title: "Track processing",
                               detail: "The Upload tab shows what's queued, processing, and ready.")
                        Bullet(systemImage: "play.rectangle.on.rectangle",
                               title: "View your gallery",
                               detail: "When a session finishes, it appears in the Gallery tab.")
                        Bullet(systemImage: "lock.fill",
                               title: "Private to you",
                               detail: "Your gallery lives at a per-user URL only you can sign into.")
                    }
                    .padding(.horizontal, 28)
                    .padding(.top, 8)

                    Text("Gallery URL: tennis.playfullife.com/u/\(userHash)")
                        .font(.caption.monospaced())
                        .foregroundColor(.secondary)
                        .padding(.top, 8)
                        .textSelection(.enabled)
                }
                .padding(.bottom, 24)
            }

            Button {
                isPresented = false
            } label: {
                Text("Get Started")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .padding(.horizontal, 24)
            .padding(.bottom, 32)
            .padding(.top, 12)
        }
    }

    private struct Bullet: View {
        let systemImage: String
        let title: String
        let detail: String

        var body: some View {
            HStack(alignment: .firstTextBaseline, spacing: 14) {
                Image(systemName: systemImage)
                    .font(.title3)
                    .foregroundColor(.accentColor)
                    .frame(width: 28)
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.subheadline.weight(.semibold))
                    Text(detail).font(.footnote).foregroundColor(.secondary)
                }
            }
        }
    }
}

/// Cheat sheet: did we already show the welcome on this device?
enum WelcomeFlag {
    private static let key = "didShowWelcomeSheet.v1"
    static var didShow: Bool {
        get { UserDefaults.standard.bool(forKey: key) }
        set { UserDefaults.standard.set(newValue, forKey: key) }
    }
}
