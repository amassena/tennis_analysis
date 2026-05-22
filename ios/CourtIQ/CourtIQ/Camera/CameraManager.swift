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
            self.captureSession.sessionPreset = .hd1920x1080

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

    /// Pick the highest-fps 1080p format. Caps at 240 fps because that
    /// is the slow-motion ceiling on current iPhones.
    private func configureBestFormat(for camera: AVCaptureDevice) {
        do {
            try camera.lockForConfiguration()
            defer { camera.unlockForConfiguration() }

            let targetFPS: Double = 240
            var bestFormat: AVCaptureDevice.Format?
            var bestFPS: Double = 0
            for format in camera.formats {
                let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                guard dims.width >= 1920 else { continue }
                for range in format.videoSupportedFrameRateRanges {
                    let maxFPS = min(range.maxFrameRate, targetFPS)
                    if maxFPS > bestFPS {
                        bestFPS = maxFPS
                        bestFormat = format
                    }
                }
            }
            if let f = bestFormat, bestFPS > 30 {
                camera.activeFormat = f
                let scale = CMTimeScale(bestFPS)
                camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: scale)
                camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: scale)
            }
            // Publish the resolved format for the live UI badge.
            let activeDims = CMVideoFormatDescriptionGetDimensions(camera.activeFormat.formatDescription)
            let activeRange = camera.activeFormat.videoSupportedFrameRateRanges.first?.maxFrameRate ?? 30
            let resolvedFPS = min(activeRange, bestFPS > 30 ? bestFPS : activeRange)
            let w = Int(activeDims.width)
            let h = Int(activeDims.height)
            DispatchQueue.main.async {
                self.activeFPS = resolvedFPS
                self.activeWidth = w
                self.activeHeight = h
            }
        } catch {
            // Not fatal — just stays at default
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
