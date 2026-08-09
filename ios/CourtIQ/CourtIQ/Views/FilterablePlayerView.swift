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
    @State private var speed: Double = 1.0          // 1.0 → 0.5 → 0.25 cycle
    @State private var usingHighFps: Bool = false    // is the current item the high-fps source?
    @State private var timeObserver: Any?
    @State private var statusObs: NSKeyValueObservation?
    @State private var srcObs: NSKeyValueObservation?  // observes a source-swap item

    /// High-fps source URL, derived from the timeline URL. Used for deep
    /// slow-mo (< ½×) so 1/4 stays smooth (240fps @ 1/4 = 60fps effective)
    /// instead of the 60fps timeline's choppy 15fps. Falls back to the
    /// timeline if this 404s (non-slo-mo capture or not-yet-backfilled).
    private var highFpsURL: URL? {
        let s = url.absoluteString
        guard s.contains("_timeline.mp4") else { return nil }
        return URL(string: s.replacingOccurrences(of: "_timeline.mp4", with: "_highfps.mp4"))
    }
    @State private var isPlaying = false
    @State private var currentTime: Double = 0
    @State private var totalDuration: Double = 0
    @State private var showControls = true
    @State private var controlsHideTask: Task<Void, Never>? = nil
    @State private var lastAutoSeekAt: TimeInterval = 0
    @State private var hasHighFps = false   // is a smooth high-fps source available?
    @State private var compareShots: Set<Int> = []   // global shot idxs with a comparison clip
    @State private var compareItem: CompareClip? = nil   // currently-presented comparison

    /// URL of the per-shot comparison clip for a global shot index,
    /// derived from the timeline URL (mirrors highFpsURL).
    private func comparisonURL(for gidx: Int) -> URL? {
        let s = url.absoluteString
        guard s.contains("_timeline.mp4") else { return nil }
        let padded = String(format: "%03d", gidx)
        return URL(string: s.replacingOccurrences(
            of: "_timeline.mp4", with: "_comparison_shot_\(padded).mp4"))
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            // Video fills the whole screen in every orientation; chrome
            // floats on top so it never steals video space. (The old VStack
            // stacked header/chips/strip above & below, which shrank the
            // video — worst in landscape where vertical room is scarce.)
            PlayerLayerContainer(player: player)
                .ignoresSafeArea()
                .contentShape(Rectangle())
                .onTapGesture { toggleControls() }

            if showControls {
                VStack(spacing: 0) {
                    // ── Top chrome: title + close + filter chips ──
                    VStack(spacing: 0) {
                        HStack {
                            Text(title.isEmpty ? (videoId ?? "") : title)
                                .font(.subheadline).foregroundColor(.white).lineLimit(1)
                            Spacer()
                            Button { dismiss() } label: {
                                Image(systemName: "xmark")
                                    .font(.title3).foregroundColor(.white).padding(8)
                            }
                        }
                        .padding(.horizontal, 14).padding(.top, 6)
                        FilterChipRow(
                            shots: shots,
                            variant: variant ?? "timeline",
                            currentFilter: $currentFilter,
                            speed: $speed,
                            onFilterChange: applyFilter,
                            onSloToggle: applySlo,
                        )
                    }
                    .background(LinearGradient(
                        colors: [Color.black.opacity(0.7), Color.black.opacity(0)],
                        startPoint: .top, endPoint: .bottom))

                    Spacer()

                    // ── Bottom chrome: playback controls + shot strip ──
                    VStack(spacing: 0) {
                        playbackControls
                        if !shots.isEmpty {
                            ShotStripRow(
                                shots: shots,
                                variant: variant ?? "timeline",
                                compareShots: compareShots,
                                onJump: jumpToShot,
                                onCompare: presentComparison,
                            )
                        }
                    }
                    .background(LinearGradient(
                        colors: [Color.black.opacity(0), Color.black.opacity(0.7)],
                        startPoint: .top, endPoint: .bottom))
                }
                .transition(.opacity)
            }
        }
        .statusBarHidden(!showControls)
        .onAppear(perform: setup)
        .onDisappear(perform: teardown)
        .fullScreenCover(item: $compareItem, onDismiss: {
            player.play()   // resume the timeline where we paused it
        }) { item in
            ComparisonClipView(url: item.url, title: item.label)
        }
    }

    /// Seek the timeline to a shot's position (in the current variant) and play.
    private func jumpToShot(_ s: PlayerShot) {
        let t = s.positions?[variant ?? "timeline"] ?? s.t
        player.seek(
            to: CMTime(seconds: max(0, t - 0.3), preferredTimescale: 600),
        ) { _ in player.play() }
    }

    /// Present a shot's side-by-side comparison clip on top of the player.
    private func presentComparison(_ s: PlayerShot) {
        guard let u = comparisonURL(for: s.idx) else { return }
        player.pause()
        compareItem = CompareClip(url: u, label: "You vs Pro — shot \(s.idx + 1)")
    }

    private var playbackControls: some View {
        VStack(spacing: 8) {
            HStack(spacing: 24) {
                Button { skip(-10) } label: {
                    Image(systemName: "gobackward.10").font(.title2).foregroundColor(.white)
                }
                Button { togglePlayPause() } label: {
                    Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                        .font(.system(size: 34, weight: .semibold))
                        .foregroundColor(.white)
                        .frame(width: 58, height: 58)
                        .background(Color.black.opacity(0.4))
                        .clipShape(Circle())
                }
                Button { skip(10) } label: {
                    Image(systemName: "goforward.10").font(.title2).foregroundColor(.white)
                }
            }
            .padding(.top, 6)

            HStack(spacing: 10) {
                Text(timeString(currentTime))
                    .font(.caption.monospacedDigit()).foregroundColor(.white)
                Slider(
                    value: Binding(
                        get: { currentTime },
                        set: { newValue in
                            currentTime = newValue
                            player.seek(to: CMTime(seconds: newValue, preferredTimescale: 600))
                        },
                    ),
                    in: 0...(max(totalDuration, 0.1)),
                )
                .tint(Color(red: 1.0, green: 0.549, blue: 0.0))
                Text(timeString(totalDuration))
                    .font(.caption.monospacedDigit()).foregroundColor(.white)
            }
            .padding(.horizontal, 14)
        }
        .padding(.top, 6)
    }

    private func togglePlayPause() {
        if isPlaying {
            player.pause()
        } else {
            player.play()
        }
        isPlaying.toggle()
        scheduleControlsHide()
    }

    private func skip(_ seconds: Double) {
        let target = max(0, currentTime + seconds)
        player.seek(to: CMTime(seconds: target, preferredTimescale: 600))
        scheduleControlsHide()
    }

    private func toggleControls() {
        withAnimation(.easeInOut(duration: 0.2)) { showControls.toggle() }
        if showControls { scheduleControlsHide() }
    }

    private func scheduleControlsHide() {
        controlsHideTask?.cancel()
        controlsHideTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(3))
            if !Task.isCancelled {
                withAnimation(.easeInOut(duration: 0.2)) { showControls = false }
            }
        }
    }


    private func timeString(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "0:00" }
        let total = Int(seconds)
        let m = total / 60
        let s = total % 60
        return String(format: "%d:%02d", m, s)
    }

    private func setup() {
        // Allow the player to rotate freely with the device (portrait or
        // landscape). The rest of the app stays portrait via the AppDelegate
        // gate; we open it up only while the player is on screen, and
        // teardown() snaps it back to portrait.
        AppDelegate.orientationLock = .allButUpsideDown
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
                totalDuration = item.duration.seconds
                if let t = startTime, t > 0 {
                    player.seek(
                        to: CMTime(seconds: t, preferredTimescale: 600),
                        toleranceBefore: .zero, toleranceAfter: .zero,
                    ) { _ in
                        player.play()
                        isPlaying = true
                        scheduleControlsHide()
                    }
                } else {
                    player.play()
                    isPlaying = true
                    scheduleControlsHide()
                }
            }
        }
        player.replaceCurrentItem(with: item)
        let shotsUrl = url.deletingLastPathComponent().appendingPathComponent("shots.json")
        fetchShots(from: shotsUrl)
        fetchComparisonManifest()
        checkHighFpsAvailable()
        let interval = CMTime(seconds: 0.25, preferredTimescale: 600)
        timeObserver = player.addPeriodicTimeObserver(
            forInterval: interval, queue: .main,
        ) { time in
            currentTime = time.seconds
            isPlaying = player.rate > 0
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
        srcObs?.invalidate()
        srcObs = nil
        controlsHideTask?.cancel()
        controlsHideTask = nil
        player.pause()
        player.replaceCurrentItem(with: nil)
        // Restore portrait lock when leaving the player so closing while
        // in fullscreen doesn't strand the rest of the app in landscape.
        if AppDelegate.orientationLock != .portrait {
            AppDelegate.orientationLock = .portrait
            if let scene = UIApplication.shared.connectedScenes
                .compactMap({ $0 as? UIWindowScene })
                .first(where: { $0.activationState == .foregroundActive }) {
                scene.requestGeometryUpdate(.iOS(interfaceOrientations: .portrait))
                scene.keyWindow?.rootViewController?
                    .setNeedsUpdateOfSupportedInterfaceOrientations()
            }
        }
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

    /// Load the per-shot comparison manifest ({vid}_comparisons_index.json)
    /// so the shot strip shows the "vs pro" affordance only where a clip
    /// exists. Best-effort; absence just means no pills.
    private func fetchComparisonManifest() {
        let base = url.deletingLastPathComponent()
        let vid = base.lastPathComponent
        let manifestURL = base.appendingPathComponent("\(vid)_comparisons_index.json")
        var req = URLRequest(url: manifestURL)
        if let jwt = TokenStore.load() {
            req.setValue("Bearer \(jwt)", forHTTPHeaderField: "Authorization")
        }
        URLSession.shared.dataTask(with: req) { data, _, _ in
            guard let data = data,
                  let resp = try? JSONDecoder().decode(ComparisonManifest.self, from: data)
            else { return }
            DispatchQueue.main.async { self.compareShots = Set(resp.shots) }
        }.resume()
    }

    /// HEAD the high-fps source to see if smooth deep slow-mo is possible
    /// for THIS video. Mixed capture formats: 120/240fps captures have one,
    /// plain 60fps captures don't — so the ¼× option is offered only when
    /// it'll actually be smooth.
    private func checkHighFpsAvailable() {
        guard let u = highFpsURL else { return }
        var req = URLRequest(url: u)
        req.httpMethod = "HEAD"
        if let jwt = TokenStore.load() {
            req.setValue("Bearer \(jwt)", forHTTPHeaderField: "Authorization")
        }
        URLSession.shared.dataTask(with: req) { _, response, _ in
            let ok = (response as? HTTPURLResponse)?.statusCode == 200
            DispatchQueue.main.async { self.hasHighFps = ok }
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

    /// Cycle the speed. With a high-fps source: 1× → ½× → ¼× → 1× (¼×
    /// switches to the high-fps source so it stays smooth). Without one
    /// (e.g. a plain 60fps capture): 1× → ½× → 1× — we skip ¼× rather than
    /// offer a choppy 15fps quarter-speed. Graceful across mixed formats.
    private func applySlo() {
        let next: Double
        if speed == 1.0 {
            next = 0.5
        } else if speed == 0.5 {
            next = hasHighFps ? 0.25 : 1.0
        } else {
            next = 1.0
        }
        speed = next
        let wantHighFps = next < 0.5
        if wantHighFps != usingHighFps {
            switchSource(highFps: wantHighFps)
        } else {
            player.play()
            player.rate = Float(next)
        }
        scheduleControlsHide()
    }

    /// Swap the AVPlayer's item between the timeline and the high-fps
    /// source, preserving the playhead. play()+rate fire only once the new
    /// item is ready (avoids the silent rate-drop race). On failure to load
    /// the high-fps item, revert to the timeline at the requested rate.
    private func switchSource(highFps: Bool) {
        let resumeAt = currentTime
        let target = highFps ? highFpsURL : url
        guard let target = target else {
            player.play(); player.rate = Float(speed); return
        }
        let item = AVPlayerItem(url: target)
        srcObs?.invalidate()
        srcObs = item.observe(\.status, options: [.new]) { item, _ in
            if item.status == .readyToPlay {
                DispatchQueue.main.async {
                    self.totalDuration = item.duration.seconds
                    self.usingHighFps = highFps
                    self.player.seek(
                        to: CMTime(seconds: resumeAt, preferredTimescale: 600),
                        toleranceBefore: .zero, toleranceAfter: .zero,
                    ) { _ in
                        self.player.play()
                        self.player.rate = Float(self.speed)
                    }
                }
            } else if item.status == .failed {
                DispatchQueue.main.async {
                    print("[FilterablePlayer] high-fps source failed; falling back to timeline")
                    if highFps {
                        // Revert to timeline, keep the slow rate (choppy but works).
                        self.usingHighFps = false
                        let fallback = AVPlayerItem(url: self.url)
                        self.player.replaceCurrentItem(with: fallback)
                        self.player.seek(to: CMTime(seconds: resumeAt, preferredTimescale: 600))
                        self.player.play()
                        self.player.rate = Float(self.speed)
                    }
                }
            }
        }
        player.replaceCurrentItem(with: item)
    }

    private func handleTimeUpdate(_ now: Double) {
        guard currentFilter != "all", let variant = variant else { return }
        // Don't auto-seek while the player is buffering / mid-seek.
        // Re-issuing seeks during stalls compounds them — that's what
        // produced the "stuck at 33s, manual skip unblocks" report.
        guard player.timeControlStatus == .playing else { return }
        let segs = buildSegments(filter: currentFilter, variant: variant)
        if segs.isEmpty { return }
        // Already inside an active segment → nothing to do.
        for s in segs where now >= s.start && now <= s.end { return }
        // Throttle auto-seeks to at most one per 1.5s. Periodic time
        // observer fires at 4 Hz; without a throttle we'd kick a second
        // seek while the first is still in flight, again causing stutter.
        let mono = ProcessInfo.processInfo.systemUptime
        guard mono - lastAutoSeekAt > 1.5 else { return }
        lastAutoSeekAt = mono
        // Tolerance: forbid backwards drift, allow forwards drift. With
        // the default (.invalid / .invalid) AVPlayer is free to snap to
        // a keyframe ON EITHER SIDE of the seek target. On videos where
        // the nearest keyframe lands a few seconds BEFORE the requested
        // time, we'd seek to 1:34.5, AVPlayer would snap to 1:32, we'd
        // see we were outside the segment again, the throttle would
        // elapse, and we'd loop forever between 1:32 and the next seek
        // attempt. toleranceBefore: .zero prevents the backwards snap.
        for s in segs where s.start > now {
            // Land INSIDE the segment. .positiveInfinity forward let
            // long-GOP/sparse-keyframe encodes jump to a far keyframe PAST the
            // shot, skipping it (#2 mis-land). Cap forward tolerance below the
            // 1.5s pre-roll so the seek decodes toward the target and lands on
            // the swing, not past it.
            let fwd = min(1.0, max(0.1, s.end - s.start - 0.5))
            player.seek(
                to: CMTime(seconds: s.start, preferredTimescale: 600),
                toleranceBefore: .zero,
                toleranceAfter: CMTime(seconds: fwd, preferredTimescale: 600),
            )
            return
        }
        // Past the last segment. Pause instead of looping back so the
        // user sees a deliberate end-of-playlist state rather than the
        // playhead snapping to the beginning.
        player.pause()
        isPlaying = false
        showControls = true
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

// Custom AVPlayerLayer-backed view (no AVPlayerViewController).
// We dropped AVPlayerViewController in build 19 because its system
// fullscreen UI took over the screen on landscape rotation, hiding the
// chip overlay AND occasionally landing in a blank/uncontrollable
// state. With our own layer + SwiftUI controls there's no fullscreen
// button at all and orientation is fully under PortraitHostingController.
struct PlayerLayerContainer: UIViewRepresentable {
    let player: AVPlayer
    func makeUIView(context: Context) -> PlayerContainerUIView {
        let v = PlayerContainerUIView()
        v.backgroundColor = .black
        v.playerLayer.player = player
        v.playerLayer.videoGravity = .resizeAspect
        return v
    }
    func updateUIView(_ uiView: PlayerContainerUIView, context: Context) {
        if uiView.playerLayer.player !== player {
            uiView.playerLayer.player = player
        }
    }
}

/// UIView whose backing layer is AVPlayerLayer. Avoids the
/// AVPlayerViewController fullscreen problem entirely.
final class PlayerContainerUIView: UIView {
    override class var layerClass: AnyClass { AVPlayerLayer.self }
    var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
}

struct FilterChipRow: View {
    let shots: [PlayerShot]
    let variant: String
    @Binding var currentFilter: String
    @Binding var speed: Double
    let onFilterChange: (String) -> Void
    let onSloToggle: () -> Void

    private var speedLabel: String {
        switch speed {
        case 0.5: return "½×"
        case 0.25: return "¼×"
        default: return "🐢 Slo"
        }
    }

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
                    label: speedLabel,
                    count: nil,
                    isActive: speed < 1.0,
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

struct ComparisonManifest: Decodable {
    let video: String
    let shots: [Int]
}

/// Identifiable wrapper so a comparison clip can drive .fullScreenCover(item:).
struct CompareClip: Identifiable {
    let id = UUID()
    let url: URL
    let label: String
}

/// Horizontal per-shot strip below the filter chips (native equivalent of
/// the web "JUMP TO SHOT" row). Tap a chip → seek to that shot. Shots that
/// have a pro comparison clip show a "vs pro" button → plays the clip.
struct ShotStripRow: View {
    let shots: [PlayerShot]
    let variant: String
    let compareShots: Set<Int>
    let onJump: (PlayerShot) -> Void
    let onCompare: (PlayerShot) -> Void

    private let typeAbbrev: [String: String] = [
        "serve": "S", "forehand": "FH", "backhand": "BH",
        "forehand_volley": "FV", "backhand_volley": "BV",
        "overhead": "OH", "unknown_shot": "?",
    ]

    private var visibleShots: [PlayerShot] {
        shots.filter { $0.positions?[variant] != nil }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("JUMP TO SHOT")
                .font(.system(size: 10, weight: .heavy))
                .tracking(0.8)
                .foregroundColor(Color(white: 0.45))
                .padding(.horizontal, 12)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(visibleShots, id: \.idx) { s in
                        VStack(spacing: 3) {
                            Text(timeLabel(s))
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundColor(Color(white: 0.6))
                            Text(typeAbbrev[s.type] ?? "?")
                                .font(.system(size: 11, weight: .bold))
                                .foregroundColor(.white)
                            if compareShots.contains(s.idx) {
                                Text("vs pro")
                                    .font(.system(size: 9, weight: .semibold))
                                    .foregroundColor(Color(red: 1.0, green: 0.55, blue: 0.0))
                                    .padding(.horizontal, 5).padding(.vertical, 2)
                                    .overlay(Capsule().stroke(
                                        Color(red: 1.0, green: 0.55, blue: 0.0).opacity(0.6),
                                        lineWidth: 1))
                                    .onTapGesture { onCompare(s) }
                            }
                        }
                        .padding(.horizontal, 9).padding(.vertical, 6)
                        .frame(minWidth: 52)
                        .background(Color(white: 0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 5))
                        .onTapGesture { onJump(s) }
                    }
                }
                .padding(.horizontal, 12)
            }
        }
        .padding(.vertical, 6)
        .background(Color.black)
    }

    private func timeLabel(_ s: PlayerShot) -> String {
        let t = Int((s.positions?[variant] ?? s.t).rounded())
        return String(format: "%d:%02d", t / 60, t % 60)
    }
}

/// Fullscreen overlay that plays a single comparison clip on top of the
/// timeline player. Closing returns to the timeline (parent resumes it via
/// the cover's onDismiss).
struct ComparisonClipView: View {
    let url: URL
    let title: String
    @Environment(\.dismiss) private var dismiss
    @State private var player = AVPlayer()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 0) {
                HStack {
                    Text(title).font(.subheadline).foregroundColor(.white).lineLimit(1)
                    Spacer()
                    Button { dismiss() } label: {
                        Image(systemName: "xmark").font(.title3).foregroundColor(.white).padding(8)
                    }
                }
                .padding(.horizontal, 14).padding(.vertical, 8)
                PlayerLayerContainer(player: player)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .onAppear {
            player.replaceCurrentItem(with: AVPlayerItem(url: url))
            player.play()
        }
        .onDisappear {
            player.pause()
            player.replaceCurrentItem(with: nil)
        }
    }
}
