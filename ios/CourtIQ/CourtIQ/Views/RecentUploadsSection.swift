import SwiftUI

/// Process-wide thumbnail cache. Survives SwiftUI view recreation so a
/// re-render (e.g. on every upload-progress update) doesn't refetch and
/// flash the placeholder.
final class ThumbnailCache {
    static let shared = ThumbnailCache()
    private let cache = NSCache<NSURL, UIImage>()
    func image(for url: URL) -> UIImage? { cache.object(forKey: url as NSURL) }
    func set(_ image: UIImage, for url: URL) { cache.setObject(image, forKey: url as NSURL) }
}

/// Thumbnail that loads once and serves from the cache thereafter, so
/// list re-renders never reflash it. Replaces AsyncImage in the upload
/// surface where progress updates re-render frequently.
struct CachedThumbnail: View {
    let url: URL
    @State private var image: UIImage?

    var body: some View {
        Group {
            if let img = image ?? ThumbnailCache.shared.image(for: url) {
                Image(uiImage: img).resizable().aspectRatio(contentMode: .fill)
            } else {
                ZStack {
                    Color.brandSurfaceElevated
                    Image(systemName: "play.rectangle")
                        .font(.system(size: 18))
                        .foregroundColor(.brandTextSecondary)
                }
                .task { await load() }
            }
        }
    }

    private func load() async {
        if ThumbnailCache.shared.image(for: url) != nil { return }
        var req = URLRequest(url: url)
        if let jwt = TokenStore.load() {
            req.setValue("Bearer \(jwt)", forHTTPHeaderField: "Authorization")
        }
        guard let (data, resp) = try? await URLSession.shared.data(for: req),
              (resp as? HTTPURLResponse)?.statusCode == 200,
              let img = UIImage(data: data) else { return }
        ThumbnailCache.shared.set(img, for: url)
        image = img
    }
}

/// "Recently uploaded" section rendered inside UploadTabView's List.
/// Pulls server-side state (post-upload-completion) so the user can see
/// "queued / processing / ready" without having to switch to the Gallery
/// tab and look for the card.
struct RecentUploadsSection: View {
    let userHash: String
    @ObservedObject var model: RecentUploadsModel
    let onTapReady: (RecentUpload) -> Void

    var body: some View {
        if !model.items.isEmpty {
            Section {
                ForEach(model.items) { item in
                    RecentRow(item: item, userHash: userHash, onTapReady: onTapReady)
                }
            } header: {
                HStack {
                    Text("Recent")
                    Spacer()
                    // Only show the spinner on the FIRST load. isLoading
                    // toggles on every background poll (every 15s), so
                    // showing it then made the header flicker during
                    // uploads. Once we have items, refreshes are silent.
                    if !model.hasLoadedOnce && model.items.isEmpty {
                        ProgressView().scaleEffect(0.6)
                    }
                }
            }
        }
    }
}

private struct RecentRow: View {
    let item: RecentUpload
    let userHash: String
    let onTapReady: (RecentUpload) -> Void

    var body: some View {
        HStack(spacing: 12) {
            thumbnail
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    statusDot
                    Text(displayName)
                        .font(.subheadline.weight(.medium))
                        .lineLimit(1)
                }
                Text(subtitle)
                    .font(.caption)
                    .foregroundColor(item.isFailed ? .red : .secondary)
                    .lineLimit(1)
            }
            Spacer()
            if item.isComplete {
                Image(systemName: "chevron.right")
                    .foregroundColor(.secondary)
                    .font(.caption.weight(.semibold))
            }
        }
        .contentShape(Rectangle())
        .onTapGesture {
            if item.isComplete { onTapReady(item) }
        }
    }

    private var thumbnail: some View {
        // CachedThumbnail (not AsyncImage): an upload-progress update
        // republishes the list and recreates these rows, and AsyncImage
        // re-fetches → flashes its placeholder each time = the flicker.
        // CachedThumbnail serves a once-loaded image from a process-wide
        // cache, so re-renders show it instantly with no reflash.
        CachedThumbnail(
            url: URL(string:
                "https://tennis.playfullife.com/u/\(userHash)/thumbs/\(item.video_id).jpg")!,
        )
        .frame(width: 64, height: 40)
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
    }

    private var displayName: String {
        let name = item.filename ?? item.video_id
        if name.count > 32 { return String(name.prefix(29)) + "…" }
        return name
    }

    private var subtitle: String {
        let when = item.uploaded_at.flatMap(Self.shortRelativeAge) ?? ""
        let stage = stageLabel
        return when.isEmpty ? stage : "\(when) · \(stage)"
    }

    private var stageLabel: String {
        if item.isComplete { return "Ready" }
        if item.isFailed { return item.error ?? "Failed" }
        switch item.status {
        case "uploading": return "Uploading\(progressSuffix)"
        case "awaiting_coordinator", "pending", "queued": return "Queued"
        case "coordinator_registered": return "Queued"
        case "downloading": return "Downloading\(progressSuffix)"
        case "preprocessing": return "Preprocessing\(progressSuffix)"
        case "extracting_poses": return "Extracting poses\(progressSuffix)"
        case "detecting_shots": return "Detecting shots\(progressSuffix)"
        case "exporting": return "Rendering clips\(progressSuffix)"
        case "uploading_results", "processing": return "Finalizing\(progressSuffix)"
        default: return item.stage ?? item.status
        }
    }

    private var progressSuffix: String {
        guard let p = item.progress, p > 0 else { return "" }
        return " · \(p)%"
    }

    private var statusDot: some View {
        Circle()
            .fill(item.isComplete ? Color.green
                  : item.isFailed ? Color.red
                  : Color.orange)
            .frame(width: 8, height: 8)
    }

    static func shortRelativeAge(from iso: String) -> String? {
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
        if seconds < 3600 { return "\(Int(seconds / 60))m ago" }
        if seconds < 86400 { return "\(Int(seconds / 3600))h ago" }
        return "\(Int(seconds / 86400))d ago"
    }
}
