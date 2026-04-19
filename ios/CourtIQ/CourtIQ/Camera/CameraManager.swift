import AVFoundation
import Vision
import UIKit

class CameraManager: NSObject, ObservableObject {
    let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private var movieOutput = AVCaptureMovieFileOutput()
    private let processingQueue = DispatchQueue(label: "com.courtiq.camera", qos: .userInitiated)

    @Published var isRecording = false

    var onFrameProcessed: (([VNHumanBodyPoseObservation]) -> Void)?

    private var frameSkipCounter = 0
    private let mlFrameInterval = 8  // process every 8th frame (30fps ML from 240fps capture)

    func configure() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1920x1080

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            print("[CameraManager] No back camera")
            return
        }

        // Configure for 240fps if available, otherwise best available
        do {
            try camera.lockForConfiguration()
            let targetFPS = 240.0
            var bestFormat: AVCaptureDevice.Format?
            var bestFrameRange: AVFrameRateRange?

            for format in camera.formats {
                let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                guard dims.width >= 1920 else { continue }
                for range in format.videoSupportedFrameRateRanges {
                    if range.maxFrameRate >= targetFPS {
                        if bestFrameRange == nil || range.maxFrameRate > bestFrameRange!.maxFrameRate {
                            bestFormat = format
                            bestFrameRange = range
                        }
                    }
                }
            }

            if let format = bestFormat, let range = bestFrameRange {
                camera.activeFormat = format
                camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: CMTimeScale(range.maxFrameRate))
                camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: CMTimeScale(range.maxFrameRate))
                print("[CameraManager] Configured at \(range.maxFrameRate) fps")
            } else {
                print("[CameraManager] 240fps not available, using default")
            }
            camera.unlockForConfiguration()
        } catch {
            print("[CameraManager] Failed to configure camera: \(error)")
        }

        guard let input = try? AVCaptureDeviceInput(device: camera) else { return }
        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
        }

        // Audio input for shot detection
        if let mic = AVCaptureDevice.default(for: .audio),
           let audioInput = try? AVCaptureDeviceInput(device: mic),
           captureSession.canAddInput(audioInput) {
            captureSession.addInput(audioInput)
        }

        // Video output for ML processing
        videoOutput.setSampleBufferDelegate(self, queue: processingQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }

        // Movie file output for recording
        if captureSession.canAddOutput(movieOutput) {
            captureSession.addOutput(movieOutput)
        }

        captureSession.commitConfiguration()
    }

    func start() {
        processingQueue.async { [weak self] in
            self?.captureSession.startRunning()
        }
    }

    func startRecording(to url: URL) {
        guard !isRecording else { return }
        movieOutput.startRecording(to: url, recordingDelegate: self)
        DispatchQueue.main.async { self.isRecording = true }
    }

    func stopRecording() {
        guard isRecording else { return }
        movieOutput.stopRecording()
    }
}

// MARK: - Video frame processing → Vision pose detection
extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        frameSkipCounter += 1
        guard frameSkipCounter % mlFrameInterval == 0 else { return }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let request = VNDetectHumanBodyPoseRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)

        do {
            try handler.perform([request])
            if let observations = request.results {
                DispatchQueue.main.async { [weak self] in
                    self?.onFrameProcessed?(observations)
                }
            }
        } catch {
            // Vision pose detection failed for this frame — skip silently
        }
    }
}

// MARK: - Recording delegate
extension CameraManager: AVCaptureFileOutputRecordingDelegate {
    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo outputFileURL: URL, from connections: [AVCaptureConnection], error: Error?) {
        DispatchQueue.main.async { self.isRecording = false }
        if let error = error {
            print("[CameraManager] Recording error: \(error.localizedDescription)")
        } else {
            print("[CameraManager] Recording saved: \(outputFileURL.lastPathComponent)")
        }
    }
}
