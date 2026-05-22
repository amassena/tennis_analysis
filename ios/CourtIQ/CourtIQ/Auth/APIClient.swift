import Foundation

/// Thin HTTP helper that knows about the tennis Worker base URL and
/// automatically attaches `Authorization: Bearer <jwt>` when a token
/// is present in `TokenStore`.
///
/// Surface stays minimal in PR 1 — we add upload-specific methods in PR 2+.
struct APIClient {
    static let baseURL = URL(string: "https://tennis.playfullife.com")!

    enum APIError: Error, LocalizedError {
        case badStatus(Int, body: String)
        case decodingFailed(underlying: Error)
        case unauthorized

        var errorDescription: String? {
            switch self {
            case .badStatus(let code, let body):
                return "HTTP \(code): \(body)"
            case .decodingFailed(let underlying):
                return "Decode failed: \(underlying.localizedDescription)"
            case .unauthorized:
                return "Not signed in (401)"
            }
        }
    }

    /// POST JSON, decode JSON response.
    static func post<RequestBody: Encodable, ResponseBody: Decodable>(
        path: String,
        body: RequestBody,
        requireAuth: Bool = true
    ) async throws -> ResponseBody {
        var req = URLRequest(url: baseURL.appendingPathComponent(path))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONEncoder().encode(body)
        try attachAuth(&req, required: requireAuth)
        return try await execute(req)
    }

    /// GET JSON.
    static func get<ResponseBody: Decodable>(
        path: String,
        requireAuth: Bool = true
    ) async throws -> ResponseBody {
        var req = URLRequest(url: baseURL.appendingPathComponent(path))
        req.httpMethod = "GET"
        try attachAuth(&req, required: requireAuth)
        return try await execute(req)
    }

    // MARK: - Internals

    private static func attachAuth(_ req: inout URLRequest, required: Bool) throws {
        if let jwt = TokenStore.load() {
            req.setValue("Bearer \(jwt)", forHTTPHeaderField: "Authorization")
        } else if required {
            throw APIError.unauthorized
        }
    }

    private static func execute<ResponseBody: Decodable>(_ req: URLRequest) async throws -> ResponseBody {
        let (data, response) = try await URLSession.shared.data(for: req)
        guard let http = response as? HTTPURLResponse else {
            throw APIError.badStatus(-1, body: "non-HTTP response")
        }
        if http.statusCode == 401 {
            throw APIError.unauthorized
        }
        guard (200..<300).contains(http.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw APIError.badStatus(http.statusCode, body: body)
        }
        do {
            return try JSONDecoder().decode(ResponseBody.self, from: data)
        } catch {
            throw APIError.decodingFailed(underlying: error)
        }
    }
}
