import SwiftUI

struct ContentView: View {
    @StateObject private var session = TennisSession()
    @State private var showingCamera = false
    @State private var showingReview = false

    var body: some View {
        ZStack {
            GalleryView()
                .ignoresSafeArea(edges: .bottom)

            VStack {
                Spacer()
                HStack {
                    Spacer()
                    Button(action: { showingCamera = true }) {
                        ZStack {
                            Circle()
                                .fill(.red)
                                .frame(width: 64, height: 64)
                                .shadow(color: .red.opacity(0.4), radius: 8)
                            Image(systemName: "video.fill")
                                .font(.system(size: 24))
                                .foregroundColor(.white)
                        }
                    }
                    .padding(.trailing, 24)
                    .padding(.bottom, 32)
                }
            }
        }
        .fullScreenCover(isPresented: $showingCamera) {
            RecordView(session: session, isPresented: $showingCamera) {
                // On recording finished → show review
                showingReview = true
            }
        }
        .sheet(isPresented: $showingReview) {
            SessionReviewView(
                recording: session.recording,
                isPresented: $showingReview
            )
        }
        .preferredColorScheme(.dark)
    }
}

// MARK: - Gallery (WebView wrapper)
struct GalleryView: View {
    var body: some View {
        WebViewWrapper(url: URL(string: "https://tennis.playfullife.com")!)
    }
}

// MARK: - Record (full-screen camera overlay)
struct RecordView: View {
    @ObservedObject var session: TennisSession
    @Binding var isPresented: Bool
    var onFinished: (() -> Void)?

    var body: some View {
        ZStack {
            CameraPreviewView(session: session)
                .ignoresSafeArea()

            PoseOverlayView(session: session)
                .ignoresSafeArea()

            VStack {
                // Top bar: close + recording info + shot counter
                HStack {
                    Button(action: {
                        if session.isRecording {
                            session.toggleRecording()
                            isPresented = false
                            onFinished?()
                        } else {
                            isPresented = false
                        }
                    }) {
                        Image(systemName: "xmark")
                            .font(.system(size: 18, weight: .bold))
                            .foregroundColor(.white)
                            .frame(width: 36, height: 36)
                            .background(.black.opacity(0.5))
                            .clipShape(Circle())
                    }

                    if session.isRecording {
                        HStack(spacing: 6) {
                            Circle()
                                .fill(.red)
                                .frame(width: 8, height: 8)
                            Text(session.recordingDuration)
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(.white)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(.black.opacity(0.5))
                        .cornerRadius(6)
                    }

                    Spacer()

                    ShotCounterView(session: session)
                }
                .padding(.horizontal, 16)
                .padding(.top, 12)

                Spacer()

                // Record button
                Button(action: { session.toggleRecording() }) {
                    ZStack {
                        Circle()
                            .stroke(.white, lineWidth: 4)
                            .frame(width: 72, height: 72)
                        if session.isRecording {
                            RoundedRectangle(cornerRadius: 4)
                                .fill(.red)
                                .frame(width: 28, height: 28)
                        } else {
                            Circle()
                                .fill(.red)
                                .frame(width: 60, height: 60)
                        }
                    }
                }
                .padding(.bottom, 40)
            }
        }
        .onAppear { session.startCamera() }
        .statusBarHidden()
    }
}

struct ShotCounterView: View {
    @ObservedObject var session: TennisSession

    var body: some View {
        HStack(spacing: 10) {
            shotPill("S", count: session.serveCount, color: .orange)
            shotPill("FH", count: session.forehandCount, color: .green)
            shotPill("BH", count: session.backhandCount, color: .blue)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(.black.opacity(0.5))
        .cornerRadius(6)
    }

    func shotPill(_ label: String, count: Int, color: Color) -> some View {
        HStack(spacing: 3) {
            Text(label)
                .font(.system(size: 10, weight: .bold))
                .foregroundColor(color)
            Text("\(count)")
                .font(.system(size: 13, weight: .bold, design: .monospaced))
                .foregroundColor(.white)
        }
    }
}
