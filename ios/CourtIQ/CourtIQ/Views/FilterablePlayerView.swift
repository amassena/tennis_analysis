import SwiftUI
import AVKit
import AVFoundation

/// Native equivalent of the web gallery's in-player chip filter.
/// Plays the timeline.mp4 with a chip row above it; tapping a chip
/// filters playback to just that shot type, auto-seeking past gaps via
/// a periodic time observer. Mirrors the web behavior 1:1 so the iOS
/// app and laptop UX stay in lockstep.
///
/// Shots are fetched lazily from `<video>/shots.json` using the user's
/// JWT (Authorization: Bearer). If the fetch fails or the variant has
/// no `positions[variant]` entries, the chip row hides and playback
/// proceeds unfiltered.
struct FilterablePlayerView: View {
    let url: URL
    let title: String
    let videoId: String?
    let variant: String?
    let startTime: Double?
    @Environment(\.dismiss) private var dismiss

    @State private var player = AVPlayer()
    @State private var shots: [PlayerShot] = []
    @State private var currentFilter: String = "all"
    @State private var sloMo: Bool = false
    @State private var timeObserver: Any?
    @State private var statusObs: NSKeyValueObservation?

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text(title.isEmpty ? (videoId ?? "") : title)
                    .font(.subheadline)
                    .foregroundColor(.white)
                    .lineLimit(1)
                Spacer()
                Button { dismiss() } label: {
                    Image(systemName: "xmark")
                        .font(.title3)
                        .foregroundColor(.white)
                        .padding(8)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(Color.black)

            // Always render the chip row so the UI is discoverable.
            // Until shots.json loads, only [All] and [Slo] are interactive;
            // type chips populate once the fetch completes.
            FilterChipRow(
                shots: shots,
                variant: variant ?? "timeline",
                currentFilter: $currentFilter,
                sloMo: $sloMo,
                onFilterChange: applyFilter,
                onSloToggle: applySlo,
            )

            AVPlayerVCContainer(player: player)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(Color.black.ignoresSafeArea())
        .onAppear(perform: setup)
        .onDisappear(perform: teardown)
    }

    private func setup() {
        let item = AVPlayerItem(url: url)
        // Calling player.play() before the item is .readyToPlay is the
        // cause of the "tap play, nothing happens, tap again, nothing,
        // third tap works" pattern reported on iphone_9ca0a615. AVPlayer
        // silently drops the rate-change request until the item finishes
        // loading. Observe status and fire play() (and the optional
        // start seek) only once the item is genuinely ready.
        statusObs = item.observe(\.status, options: [.new]) { item, _ in
            guard item.status == .readyToPlay else { return }
            DispatchQueue.main.async {
                if let t = startTime, t > 0 {
                    player.seek(
                        to: CMTime(seconds: t, preferredTimescale: 600),
                        toleranceBefore: .zero, toleranceAfter: .zero,
                    ) { _ in player.play() }
                } else {
                    player.play()
                }
            }
        }
        player.replaceCurrentItem(with: item)
        let shotsUrl = url.deletingLastPathComponent().appendingPathComponent("shots.json")
        fetchShots(from: shotsUrl)
        let interval = CMTime(seconds: 0.25, preferredTimescale: 600)
        timeObserver = player.addPeriodicTimeObserver(
            forInterval: interval, queue: .main,
        ) { time in
            handleTimeUpdate(time.seconds)
        }
    }

    private func teardown() {
        if let obs = timeObserver {
            player.removeTimeObserver(obs)
            timeObserver = nil
        }
        statusObs?.invalidate()
        statusObs = nil
        player.pause()
        player.replaceCurrentItem(with: nil)
    }

    private func fetchShots(from url: URL) {
        var req = URLRequest(url: url)
        if let jwt = TokenStore.load() {
            req.setValue("Bearer \(jwt)", forHTTPHeaderField: "Authorization")
        } else {
            print("[FilterablePlayer] no JWT to attach to shots fetch")
        }
        URLSession.shared.dataTask(with: req) { data, response, error in
            if let error = error {
                print("[FilterablePlayer] shots fetch error: \(error)")
                return
            }
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            guard let data = data else {
                print("[FilterablePlayer] shots fetch returned no data (status \(status))")
                return
            }
            do {
                let resp = try JSONDecoder().decode(ShotsResponse.self, from: data)
                print("[FilterablePlayer] loaded \(resp.shots.count) shots for \(resp.video) (status \(status))")
                DispatchQueue.main.async { self.shots = resp.shots }
            } catch {
                let preview = String(data: data.prefix(120), encoding: .utf8) ?? "<binary>"
                print("[FilterablePlayer] decode failed (status \(status)): \(error) — body[0..120]=\(preview)")
            }
        }.resume()
    }

    private func applyFilter(_ f: String) {
        currentFilter = f
        guard let variant = variant else { return }
        let segs = buildSegments(filter: f, variant: variant)
        if let first = segs.first {
            // Use the completion-handler seek so play() only fires once
            // the playhead is actually at the target. Plain seek+play()
            // back-to-back races and AVPlayer silently drops the rate
            // change while the seek is in flight, which produced the
            // "need to tap play 3 times to start" report.
            player.seek(
                to: CMTime(seconds: first.start, preferredTimescale: 600),
                toleranceBefore: .zero, toleranceAfter: .zero,
            ) { _ in player.play() }
        }
    }

    private func applySlo() {
        sloMo.toggle()
        // Setting rate alone doesn't always stick if the player is mid-pause;
        // play() then set rate is the documented way.
        if sloMo {
            player.play()
            player.rate = 0.5
        } else {
            player.rate = 1.0
        }
    }

    private func handleTimeUpdate(_ now: Double) {
        guard currentFilter != "all", let variant = variant else { return }
        let segs = buildSegments(filter: currentFilter, variant: variant)
        if segs.isEmpty { return }
        for s in segs where now >= s.start && now <= s.end { return }
        for s in segs where s.start > now {
            player.seek(
                to: CMTime(seconds: s.start, preferredTimescale: 600),
                toleranceBefore: .zero, toleranceAfter: .zero,
            )
            return
        }
        if let first = segs.first {
            player.seek(
                to: CMTime(seconds: first.start, preferredTimescale: 600),
                toleranceBefore: .zero, toleranceAfter: .zero,
            )
        }
    }

    private func buildSegments(filter: String, variant: String) -> [(start: Double, end: Double)] {
        if filter == "rally" { return buildRallySegments(variant: variant) }
        let types = PlayerFilter.typeMap[filter]
        var raw: [(Double, Double)] = []
        for s in shots {
            guard let pos = s.positions?[variant] else { continue }
            if let types = types, !types.contains(s.type) { continue }
            raw.append((max(0, pos - 1.5), pos + 2.5))
        }
        raw.sort { $0.0 < $1.0 }
        var merged: [(Double, Double)] = []
        for s in raw {
            if let last = merged.last, s.0 <= last.1 + 0.3 {
                merged[merged.count - 1].1 = max(last.1, s.1)
            } else {
                merged.append(s)
            }
        }
        return merged.map { (start: $0.0, end: $0.1) }
    }

    /// Rally = shots within 8s of each other grouped into one continuous
    /// "point" segment (3.5s before first shot through 4.5s after last).
    /// Matches the server-side rally.mp4 generation algorithm, derived
    /// here so the GPU pipeline can stop emitting rally.mp4 entirely.
    func buildRallySegments(variant: String) -> [(start: Double, end: Double)] {
        let times = shots.compactMap { $0.positions?[variant] }.sorted()
        guard !times.isEmpty else { return [] }
        let pointGap = 8.0, before = 3.5, after = 4.5
        var points: [[Double]] = [[times[0]]]
        for t in times.dropFirst() {
            if let last = points.last?.last, t - last > pointGap {
                points.append([t])
            } else {
                points[points.count - 1].append(t)
            }
        }
        return points.map { p in
            (start: max(0, p.first! - before), end: p.last! + after)
        }
    }
}

// Wraps AVPlayerViewController so we keep Apple's playback chrome
// (scrubber, AirPlay, PiP) while owning the layout around it.
struct AVPlayerVCContainer: UIViewControllerRepresentable {
    let player: AVPlayer
    func makeUIViewController(context: Context) -> AVPlayerViewController {
        let vc = AVPlayerViewController()
        vc.player = player
        vc.allowsPictureInPicturePlayback = true
        vc.entersFullScreenWhenPlaybackBegins = false
        vc.view.backgroundColor = .black
        return vc
    }
    func updateUIViewController(_ vc: AVPlayerViewController, context: Context) {}
}

struct FilterChipRow: View {
    let shots: [PlayerShot]
    let variant: String
    @Binding var currentFilter: String
    @Binding var sloMo: Bool
    let onFilterChange: (String) -> Void
    let onSloToggle: () -> Void

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                chip(label: "All", count: totalCount, isActive: currentFilter == "all") {
                    onFilterChange("all")
                }
                let rallyCount = PlayerFilter.rallyPointCount(shots: shots, variant: variant)
                if rallyCount > 0 {
                    chip(label: "Rally", count: rallyCount, isActive: currentFilter == "rally") {
                        onFilterChange("rally")
                    }
                }
                ForEach(PlayerFilter.categories, id: \.key) { cat in
                    let c = countFor(cat.key)
                    if c > 0 {
                        chip(label: cat.label, count: c, isActive: currentFilter == cat.key) {
                            onFilterChange(cat.key)
                        }
                    }
                }
                Spacer(minLength: 12)
                chip(
                    label: "🐢 Slo",
                    count: nil,
                    isActive: sloMo,
                    activeColor: Color(red: 0.608, green: 0.349, blue: 0.714),
                ) {
                    onSloToggle()
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
        .background(Color.black)
    }

    private var totalCount: Int {
        shots.filter { $0.positions?[variant] != nil }.count
    }

    private func countFor(_ key: String) -> Int {
        let types = PlayerFilter.typeMap[key] ?? []
        return shots.filter {
            $0.positions?[variant] != nil && types.contains($0.type)
        }.count
    }

    @ViewBuilder
    private func chip(
        label: String,
        count: Int?,
        isActive: Bool,
        activeColor: Color = Color(red: 1.0, green: 0.549, blue: 0.0),
        action: @escaping () -> Void,
    ) -> some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Text(label)
                    .font(.system(size: 13, weight: .semibold))
                if let c = count {
                    Text("\(c)")
                        .font(.system(size: 11, weight: .medium))
                        .opacity(isActive ? 0.95 : 0.7)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(isActive ? activeColor : Color(white: 0.1))
            .foregroundColor(isActive ? .black : Color(white: 0.7))
            .clipShape(Capsule())
            .overlay(
                Capsule().stroke(
                    isActive ? activeColor : Color(white: 0.18), lineWidth: 1,
                ),
            )
        }
        .buttonStyle(.plain)
    }
}

enum PlayerFilter {
    static let typeMap: [String: [String]] = [
        "serve":    ["serve"],
        "forehand": ["forehand"],
        "backhand": ["backhand"],
        "volley":   ["forehand_volley", "backhand_volley"],
        "overhead": ["overhead"],
    ]
    static let categories: [(key: String, label: String)] = [
        ("serve",    "Serve"),
        ("forehand", "FH"),
        ("backhand", "BH"),
        ("volley",   "Volley"),
        ("overhead", "OH"),
    ]

    /// Rally segment count for the given variant — used to decide whether
    /// to render the Rally chip and what number to display.
    static func rallyPointCount(shots: [PlayerShot], variant: String) -> Int {
        let times = shots.compactMap { $0.positions?[variant] }.sorted()
        guard !times.isEmpty else { return 0 }
        var count = 1
        for i in 1..<times.count {
            if times[i] - times[i - 1] > 8.0 { count += 1 }
        }
        return count
    }
}

struct PlayerShot: Decodable {
    let idx: Int
    let t: Double
    let type: String
    let positions: [String: Double]?
}

struct ShotsResponse: Decodable {
    let video: String
    let shots: [PlayerShot]
}
