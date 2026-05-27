import SwiftUI
import AVFoundation

/// Full-screen camera UI: tap to record, tap to stop, confirm → upload.
struct RecordView: View {
    let userHash: String
    @Binding var isPresented: Bool

    @StateObject private var camera = CameraManager()
    @State private var recordedURL: URL?

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if camera.permissionGranted == false {
                PermissionDeniedView { isPresented = false }
            } else if let url = recordedURL {
                ReviewView(
                    recordedURL: url,
                    userHash: userHash,
                    onUpload: {
                        UploadManager.shared.enqueue(
                            localFileURL: url,
                            assetId: "\(userHash)_\(UUID().uuidString)",
                            filename: url.lastPathComponent,
                            userHash: userHash
                        )
                        isPresented = false
                    },
                    onDiscard: {
                        try? FileManager.default.removeItem(at: url)
                        recordedURL = nil
                    }
                )
            } else {
                LiveCameraView(camera: camera) {
                    isPresented = false
                } onStopped: { url in
                    recordedURL = url
                }
            }
        }
        .onAppear {
            camera.requestPermissionAndConfigure()
        }
        .onDisappear {
            camera.teardown()
        }
        .statusBarHidden()
    }
}

// MARK: - Live camera

private struct LiveCameraView: View {
    @ObservedObject var camera: CameraManager
    var onClose: () -> Void
    var onStopped: (URL) -> Void

    var body: some View {
        ZStack {
            CameraPreviewView(session: camera.captureSession)
                .ignoresSafeArea()

            // Rule-of-thirds grid overlay (helps frame the player)
            GridOverlay()
                .ignoresSafeArea()
                .allowsHitTesting(false)

            VStack {
                HStack {
                    Button(action: onClose) {
                        Image(systemName: "xmark")
                            .font(.system(size: 18, weight: .bold))
                            .foregroundColor(.white)
                            .frame(width: 36, height: 36)
                            .background(Color.black.opacity(0.5))
                            .clipShape(Circle())
                    }
                    if camera.isRecording {
                        HStack(spacing: 6) {
                            Circle().fill(Color.red).frame(width: 8, height: 8)
                            Text(formatDuration(camera.recordingDuration))
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(.white)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(Color.black.opacity(0.5))
                        .cornerRadius(6)
                    }
                    Spacer()
                    // Quality badge: e.g. "1080p · 240 fps" — confirms the
                    // capture format we picked.
                    if camera.activeFPS > 0 && camera.activeHeight > 0 {
                        Text(qualityText)
                            .font(.system(.caption2, design: .monospaced))
                            .foregroundColor(.white)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.black.opacity(0.5))
                            .cornerRadius(6)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.top, 12)

                Spacer()

                Button {
                    if camera.isRecording {
                        camera.stopRecording()
                    } else {
                        camera.startRecording { result in
                            if case .success(let url) = result {
                                onStopped(url)
                            }
                        }
                    }
                } label: {
                    ZStack {
                        Circle()
                            .stroke(Color.white, lineWidth: 4)
                            .frame(width: 78, height: 78)
                        if camera.isRecording {
                            RoundedRectangle(cornerRadius: 4)
                                .fill(Color.red)
                                .frame(width: 30, height: 30)
                        } else {
                            Circle()
                                .fill(Color.red)
                                .frame(width: 64, height: 64)
                        }
                    }
                }
                .padding(.bottom, 40)
            }
        }
    }

    private func formatDuration(_ t: TimeInterval) -> String {
        let mins = Int(t) / 60
        let secs = Int(t) % 60
        let tenths = Int((t - floor(t)) * 10)
        return String(format: "%d:%02d.%d", mins, secs, tenths)
    }

    private var qualityText: String {
        let shortDim = min(camera.activeWidth, camera.activeHeight)
        let resLabel: String
        switch shortDim {
        case 2160...: resLabel = "4K"
        case 1440...: resLabel = "1440p"
        case 1080...: resLabel = "1080p"
        case 720...: resLabel = "720p"
        default: resLabel = "\(shortDim)p"
        }
        return "\(resLabel) · \(Int(camera.activeFPS)) fps"
    }
}

// Rule-of-thirds grid overlay — two horizontal + two vertical
// lines at the 1/3 and 2/3 marks. Light white at 30% opacity so it's
// visible but doesn't dominate the frame.
private struct GridOverlay: View {
    var body: some View {
        GeometryReader { geo in
            Path { path in
                let w = geo.size.width
                let h = geo.size.height
                let x1 = w / 3
                let x2 = 2 * w / 3
                let y1 = h / 3
                let y2 = 2 * h / 3
                path.move(to: CGPoint(x: x1, y: 0)); path.addLine(to: CGPoint(x: x1, y: h))
                path.move(to: CGPoint(x: x2, y: 0)); path.addLine(to: CGPoint(x: x2, y: h))
                path.move(to: CGPoint(x: 0, y: y1)); path.addLine(to: CGPoint(x: w, y: y1))
                path.move(to: CGPoint(x: 0, y: y2)); path.addLine(to: CGPoint(x: w, y: y2))
            }
            .stroke(Color.white.opacity(0.28), lineWidth: 0.5)
        }
    }
}

// MARK: - Confirmation after recording

private struct ReviewView: View {
    let recordedURL: URL
    let userHash: String
    var onUpload: () -> Void
    var onDiscard: () -> Void

    @State private var player: AVPlayer?

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 0) {
                if let player = player {
                    VideoPlayerWrapper(player: player)
                        .aspectRatio(9.0/16.0, contentMode: .fit)
                        .frame(maxWidth: .infinity)
                } else {
                    ProgressView().tint(.white)
                }
                Spacer()
                VStack(spacing: 12) {
                    Button(action: onUpload) {
                        Label("Upload", systemImage: "arrow.up.circle.fill")
                            .font(.title3.weight(.semibold))
                            .frame(maxWidth: .infinity, minHeight: 50)
                    }
                    .buttonStyle(.borderedProminent)
                    Button(role: .destructive, action: onDiscard) {
                        Label("Discard and re-record", systemImage: "trash")
                            .frame(maxWidth: .infinity, minHeight: 44)
                    }
                    .buttonStyle(.bordered)
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 36)
            }
        }
        .onAppear {
            player = AVPlayer(url: recordedURL)
            player?.play()
        }
        .onDisappear { player?.pause() }
    }
}

// MARK: - Permission denied

private struct PermissionDeniedView: View {
    var onClose: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "video.slash")
                .font(.system(size: 56))
                .foregroundColor(.white)
            Text("Camera access needed")
                .font(.title3.weight(.semibold))
                .foregroundColor(.white)
            Text("Enable camera access in Settings to record videos.")
                .font(.subheadline)
                .foregroundColor(.white.opacity(0.7))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 36)
            HStack(spacing: 12) {
                Button("Open Settings") {
                    // Deep-links into the Tennis Uploader entry in iOS Settings.
                    if let url = URL(string: UIApplication.openSettingsURLString) {
                        UIApplication.shared.open(url)
                    }
                }
                .buttonStyle(.borderedProminent)
                Button("Close", action: onClose)
                    .buttonStyle(.bordered)
                    .tint(.white)
            }
            .padding(.top, 12)
        }
    }
}

// MARK: - AVPlayer wrapper (replace VideoPlayer to avoid macOS-only imports in tests)

private struct VideoPlayerWrapper: UIViewControllerRepresentable {
    let player: AVPlayer

    func makeUIViewController(context: Context) -> AVPlayerViewControllerWrapper {
        let vc = AVPlayerViewControllerWrapper()
        vc.attach(player: player)
        return vc
    }

    func updateUIViewController(_ uiViewController: AVPlayerViewControllerWrapper, context: Context) {}
}

private final class AVPlayerViewControllerWrapper: UIViewController {
    func attach(player: AVPlayer) {
        let layer = AVPlayerLayer(player: player)
        layer.videoGravity = .resizeAspect
        layer.frame = view.bounds
        layer.backgroundColor = UIColor.black.cgColor
        view.layer.addSublayer(layer)
        self.playerLayer = layer
    }
    private var playerLayer: AVPlayerLayer?
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        playerLayer?.frame = view.bounds
    }
}
