import Foundation
import AVFoundation
import Combine

/// Recording-only camera. No Vision / live-coaching work — that will
/// come from the live-coaching stream in a future PR and likely lives
/// in a separate manager class.
///
/// Tries to configure 240 fps at 1080p if available (matches the
/// product 240fps slo-mo bar); falls back to the best available.
final class CameraManager: NSObject, ObservableObject {
    let captureSession = AVCaptureSession()
    private let movieOutput = AVCaptureMovieFileOutput()
    private let sessionQueue = DispatchQueue(label: "com.playfullife.courtiq.camera")

    @Published var isRecording = false
    @Published var recordingDuration: TimeInterval = 0
    @Published var lastError: String?
    @Published var lastRecordingURL: URL?
    @Published var permissionGranted: Bool? = nil  // nil = unknown
    /// Resolved capture format, shown in the live camera UI as a small badge.
    @Published var activeFPS: Double = 0
    @Published var activeWidth: Int = 0
    @Published var activeHeight: Int = 0
    /// All capture qualities offered by the back camera. Populated once
    /// during configure() so Settings can render a picker.
    @Published var availableQualities: [VideoQuality] = []

    private var startedAt: Date?
    private var timer: AnyCancellable?
    private var onFinished: ((Result<URL, Error>) -> Void)?

    func requestPermissionAndConfigure() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] vGranted in
            AVCaptureDevice.requestAccess(for: .audio) { aGranted in
                DispatchQueue.main.async {
                    self?.permissionGranted = vGranted
                }
                if vGranted {
                    self?.configure(audioGranted: aGranted)
                }
            }
        }
    }

    private func configure(audioGranted: Bool) {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.captureSession.beginConfiguration()
            // .high is a sensible fallback; configureBestFormat will set
            // activeFormat below, which switches the session into
            // inputPriority mode and overrides this preset.
            self.captureSession.sessionPreset = .high

            guard
                let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                let videoInput = try? AVCaptureDeviceInput(device: camera)
            else {
                DispatchQueue.main.async { self.lastError = "Back camera unavailable" }
                self.captureSession.commitConfiguration()
                return
            }

            self.configureBestFormat(for: camera)

            if self.captureSession.canAddInput(videoInput) {
                self.captureSession.addInput(videoInput)
            }

            if audioGranted,
               let mic = AVCaptureDevice.default(for: .audio),
               let audioInput = try? AVCaptureDeviceInput(device: mic),
               self.captureSession.canAddInput(audioInput) {
                self.captureSession.addInput(audioInput)
            }

            if self.captureSession.canAddOutput(self.movieOutput) {
                self.captureSession.addOutput(self.movieOutput)
            }

            self.captureSession.commitConfiguration()
            self.captureSession.startRunning()
        }
    }

    /// Enumerate the device's offered (resolution, fps) combinations and
    /// honor the user's stored preference. Two paths:
    ///
    ///   • Auto: floor at 60 fps, maximize resolution, take the lowest fps
    ///     ≥60 at that resolution (so we get 4K · 60 on iPhone 16 Pro
    ///     rather than 4K · 120 if it existed). Default behavior.
    ///   • Manual: SettingsView writes a VideoQuality.id (e.g.
    ///     "3840x2160@60") to UserDefaults; we honor exactly that.
    ///
    /// Either way, publishes `availableQualities` for the Settings picker
    /// and `active{FPS,Width,Height}` for the live camera badge.
    private func configureBestFormat(for camera: AVCaptureDevice) {
        do {
            try camera.lockForConfiguration()
            defer { camera.unlockForConfiguration() }

            // Build the catalog of every (width, height, fps) combo
            // the device offers. Dedupe — Apple often exposes multiple
            // formats with identical (res, fps) but different codecs.
            var catalog: [VideoQuality: AVCaptureDevice.Format] = [:]
            for format in camera.formats {
                let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                let w = Int(dims.width)
                let h = Int(dims.height)
                let maxFPS = format.videoSupportedFrameRateRanges
                    .map { $0.maxFrameRate }.max() ?? 0
                guard maxFPS > 0 else { continue }
                let q = VideoQuality(width: w, height: h, fps: Int(maxFPS))
                // First write wins; later duplicates ignored.
                if catalog[q] == nil { catalog[q] = format }
            }

            // Publish sorted descending by resolution then fps so Settings
            // shows the most capable options first.
            let sortedQualities = catalog.keys.sorted { lhs, rhs in
                let lp = lhs.width * lhs.height
                let rp = rhs.width * rhs.height
                if lp != rp { return lp > rp }
                return lhs.fps > rhs.fps
            }
            DispatchQueue.main.async {
                self.availableQualities = sortedQualities
            }

            // Decide which entry to apply.
            let prefId = UserDefaults.standard.string(forKey: "videoQualityPref") ?? "auto"
            let pick: (q: VideoQuality, format: AVCaptureDevice.Format)? = {
                if prefId != "auto", let preferred = sortedQualities.first(where: { $0.id == prefId }),
                   let f = catalog[preferred] {
                    return (preferred, f)
                }
                // Auto: floor at 60 fps, max resolution, HIGHEST fps at that res.
                // On iPhone 16 Pro this resolves to 4K · 120 fps (not 4K · 60),
                // because once we're committing to 4K we may as well take the
                // most temporal detail the device offers.
                let eligible = sortedQualities.filter { $0.fps >= 60 }
                guard let maxRes = eligible.first.map({ $0.width * $0.height }) else { return nil }
                let atMaxRes = eligible.filter { $0.width * $0.height == maxRes }
                guard let chosen = atMaxRes.max(by: { $0.fps < $1.fps }),
                      let f = catalog[chosen] else { return nil }
                return (chosen, f)
            }()

            guard let pick else { return }

            camera.activeFormat = pick.format
            let scale = CMTimeScale(pick.q.fps)
            camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: scale)
            camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: scale)

            let q = pick.q
            DispatchQueue.main.async {
                self.activeFPS = Double(q.fps)
                self.activeWidth = q.width
                self.activeHeight = q.height
            }
        } catch {
            // Not fatal — just stays at default
        }
    }

    /// Settings view enumerates capture qualities without starting a session.
    /// Reads back-camera formats once, dedupes, returns sorted (res desc, fps desc).
    static func enumerateAvailableQualities() -> [VideoQuality] {
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
        else { return [] }
        var seen = Set<VideoQuality>()
        for format in camera.formats {
            let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            let maxFPS = format.videoSupportedFrameRateRanges
                .map { $0.maxFrameRate }.max() ?? 0
            guard maxFPS > 0 else { continue }
            seen.insert(VideoQuality(width: Int(dims.width), height: Int(dims.height), fps: Int(maxFPS)))
        }
        return seen.sorted { lhs, rhs in
            let lp = lhs.width * lhs.height
            let rp = rhs.width * rhs.height
            if lp != rp { return lp > rp }
            return lhs.fps > rhs.fps
        }
    }

    /// Re-apply the user's quality preference (call after Settings changes it).
    /// Safe to invoke while the session is running.
    func reapplyQualityPreference() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            guard let camera = (self.captureSession.inputs.compactMap { $0 as? AVCaptureDeviceInput }
                .first { $0.device.hasMediaType(.video) }?.device) else { return }
            self.captureSession.beginConfiguration()
            self.configureBestFormat(for: camera)
            self.captureSession.commitConfiguration()
        }
    }

    /// Start recording to a freshly-staged file under `UploadStaging`.
    func startRecording(onFinished: @escaping (Result<URL, Error>) -> Void) {
        guard !isRecording else { return }
        self.onFinished = onFinished
        let url = UploadStaging.stagingURL(for: "rec_\(Int(Date().timeIntervalSince1970)).mov")
        DispatchQueue.main.async {
            self.startedAt = Date()
            self.recordingDuration = 0
            self.isRecording = true
            self.timer = Timer.publish(every: 0.1, on: .main, in: .common)
                .autoconnect()
                .sink { [weak self] _ in
                    guard let self, let s = self.startedAt else { return }
                    self.recordingDuration = Date().timeIntervalSince(s)
                }
        }
        sessionQueue.async {
            self.movieOutput.startRecording(to: url, recordingDelegate: self)
        }
    }

    func stopRecording() {
        guard isRecording else { return }
        sessionQueue.async {
            self.movieOutput.stopRecording()
        }
    }

    func teardown() {
        sessionQueue.async {
            self.captureSession.stopRunning()
        }
        timer?.cancel()
        timer = nil
    }
}

/// One (resolution, fps) capture option offered by the device.
/// Used both as a Settings menu item and as the value persisted in
/// UserDefaults under key "videoQualityPref" (or "auto").
struct VideoQuality: Hashable, Identifiable {
    let width: Int
    let height: Int
    let fps: Int

    var id: String { "\(width)x\(height)@\(fps)" }

    var displayName: String {
        let shortDim = min(width, height)
        let resLabel: String
        switch shortDim {
        case 2160...: resLabel = "4K"
        case 1440...: resLabel = "1440p"
        case 1080...: resLabel = "1080p"
        case 720...: resLabel = "720p"
        default: resLabel = "\(shortDim)p"
        }
        return "\(resLabel) · \(fps) fps"
    }
}

extension CameraManager: AVCaptureFileOutputRecordingDelegate {
    func fileOutput(
        _ output: AVCaptureFileOutput,
        didFinishRecordingTo outputFileURL: URL,
        from connections: [AVCaptureConnection],
        error: Error?
    ) {
        DispatchQueue.main.async {
            self.isRecording = false
            self.timer?.cancel()
            self.timer = nil
            if let error {
                self.lastError = error.localizedDescription
                self.onFinished?(.failure(error))
            } else {
                self.lastRecordingURL = outputFileURL
                self.onFinished?(.success(outputFileURL))
            }
            self.onFinished = nil
        }
    }
}
