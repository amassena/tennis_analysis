import SwiftUI
import AVKit

/// "Today's session" hero card — pinned to the top of the Upload tab.
/// Shows the most recently-ready video as a big thumbnail with one-tap
/// playback via native AVPlayerViewController. Skips entirely when the
/// user has no ready videos yet (the EmptyUploadsView handles that case).
///
/// Driven by the same RecentUploadsModel that powers the Recent section,
/// so we don't double-poll the worker.
struct TodayHeroCard: View {
    let userHash: String
    @ObservedObject var recent: RecentUploadsModel

    @State private var presentingPlayer: PresentablePlayer?

    var body: some View {
        if let item = latestReadyItem {
            heroButton(for: item)
        } else if recent.items.isEmpty && recent.isLoading {
            loadingPlaceholder
        } else if recent.items.contains(where: { !$0.isComplete }) {
            // Recent has data but no completed videos yet — show
            // "processing" placeholder rather than nothing.
            processingPlaceholder
        }
    }

    @ViewBuilder
    private func heroButton(for item: RecentUpload) -> some View {
        Button {
            presentingPlayer = PresentablePlayer(
                url: timelineURL(for: item),
                title: item.filename ?? item.video_id,
            )
        } label: {
                VStack(alignment: .leading, spacing: 0) {
                    thumbnail(for: item)
                        .overlay(alignment: .topLeading) { tagPill }
                        .overlay(alignment: .center) { playGlyph }
                        .overlay(alignment: .bottomLeading) { caption(item) }

                    HStack(spacing: 10) {
                        if let when = relativeAge(for: item) {
                            Text(when)
                                .font(.subheadline.weight(.medium))
                                .foregroundColor(.brandText)
                        }
                        Spacer()
                        Image(systemName: "arrow.up.right.square")
                            .foregroundColor(.brandTextSecondary)
                            .font(.subheadline)
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 12)
                }
                .background(Color.brandSurface)
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            }
            .buttonStyle(.plain)
            .listRowInsets(EdgeInsets(top: 12, leading: 16, bottom: 8, trailing: 16))
            .listRowSeparator(.hidden)
            .listRowBackground(Color.clear)
            .fullScreenCover(item: $presentingPlayer) { p in
                NativePlayerView(url: p.url, title: p.title)
            }
        }
    }

    private var latestReadyItem: RecentUpload? {
        recent.items.first(where: { $0.isComplete })
    }

    private var loadingPlaceholder: some View {
        HStack {
            ProgressView().tint(.brandAccent)
            Text("Loading your sessions…")
                .font(.subheadline)
                .foregroundColor(.brandTextSecondary)
            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 28)
        .frame(maxWidth: .infinity)
        .background(Color.brandSurface)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .listRowInsets(EdgeInsets(top: 12, leading: 16, bottom: 8, trailing: 16))
        .listRowSeparator(.hidden)
        .listRowBackground(Color.clear)
    }

    private var processingPlaceholder: some View {
        VStack(spacing: 8) {
            Image(systemName: "gearshape.2")
                .font(.system(size: 36))
                .foregroundColor(.brandAccent)
            Text("Your first session is processing")
                .font(.subheadline.weight(.medium))
                .foregroundColor(.brandText)
            Text("We'll show it here once the GPU is done.")
                .font(.caption)
                .foregroundColor(.brandTextSecondary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 22)
        .frame(maxWidth: .infinity)
        .background(Color.brandSurface)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .listRowInsets(EdgeInsets(top: 12, leading: 16, bottom: 8, trailing: 16))
        .listRowSeparator(.hidden)
        .listRowBackground(Color.clear)
    }

    private func thumbnail(for item: RecentUpload) -> some View {
        let url = URL(string: "https://tennis.playfullife.com/u/\(userHash)/thumbs/\(item.video_id).jpg")!
        return AsyncImage(url: url) { phase in
            switch phase {
            case .success(let image):
                image.resizable().aspectRatio(contentMode: .fill)
            default:
                ZStack {
                    Color.brandSurfaceElevated
                    Image(systemName: "play.rectangle.fill")
                        .font(.system(size: 44))
                        .foregroundColor(.brandTextSecondary)
                }
            }
        }
        .frame(maxWidth: .infinity)
        .frame(height: 200)
        .clipped()
    }

    private var tagPill: some View {
        Text("TODAY")
            .font(.caption2.weight(.heavy))
            .tracking(0.8)
            .foregroundColor(.brandBackground)
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(Color.brandAccent)
            .clipShape(Capsule())
            .padding(12)
    }

    private var playGlyph: some View {
        ZStack {
            Circle().fill(.black.opacity(0.45))
            Image(systemName: "play.fill")
                .font(.system(size: 28, weight: .bold))
                .foregroundColor(.brandText)
        }
        .frame(width: 64, height: 64)
    }

    private func caption(_ item: RecentUpload) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(item.filename ?? item.video_id)
                .font(.headline)
                .foregroundColor(.brandText)
                .lineLimit(1)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(
            LinearGradient(
                colors: [.black.opacity(0.6), .black.opacity(0)],
                startPoint: .bottom, endPoint: .top,
            )
        )
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func relativeAge(for item: RecentUpload) -> String? {
        guard let iso = item.uploaded_at else { return nil }
        return RecentRow_shortRelativeAge(iso)
    }

    private func timelineURL(for item: RecentUpload) -> URL {
        // The legacy export naming is <vid>_timeline.mp4. Worker per-user
        // JWT fallback resolves it under highlights/<hash>/<vid>/.
        URL(string: "https://tennis.playfullife.com/u/\(userHash)/\(item.video_id)/\(item.video_id)_timeline.mp4")!
    }
}

/// Shared relative-age helper. Mirrors RecentRow.shortRelativeAge so we
/// don't reach into a private type.
func RecentRow_shortRelativeAge(_ iso: String) -> String? {
    let f = ISO8601DateFormatter()
    f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    var date = f.date(from: iso)
    if date == nil {
        f.formatOptions = [.withInternetDateTime]
        date = f.date(from: iso)
    }
    guard let d = date else { return nil }
    let seconds = -d.timeIntervalSinceNow
    if seconds < 60 { return "just now" }
    if seconds < 3600 { return "\(Int(seconds / 60)) min ago" }
    if seconds < 86400 { return "\(Int(seconds / 3600)) hr ago" }
    return "\(Int(seconds / 86400)) days ago"
}

struct PresentablePlayer: Identifiable {
    let id = UUID()
    let url: URL
    let title: String
}

struct NativePlayerView: UIViewControllerRepresentable {
    let url: URL
    let title: String

    func makeUIViewController(context: Context) -> AVPlayerViewController {
        let player = AVPlayer(url: url)
        let vc = AVPlayerViewController()
        vc.player = player
        vc.allowsPictureInPicturePlayback = true
        DispatchQueue.main.async { player.play() }
        return vc
    }

    func updateUIViewController(_ uiViewController: AVPlayerViewController, context: Context) {}
}
