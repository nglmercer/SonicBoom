/**
 * Options for synthesizing and playing audio on the server.
 */
export interface TtsPlayOptions {
  /** Voice style (e.g., 'M1', 'F1', 'M2', etc.) */
  voice?: string;
  /** Language code (default: 'en') */
  lang?: string;
  /** If true, clears the queue and plays immediately. If false, adds to the queue. */
  playNow?: boolean;
}

/**
 * Response returned by the SonicBoom /api/tts/play API.
 */
export interface TtsPlayResponse {
  success: boolean;
  message: string;
  id: string;
}

/**
 * Synthesizes text and plays it directly on the SonicBoom server's audio output.
 *
 * @param baseUrl - The base URL of the SonicBoom server (e.g., 'http://localhost:3000')
 * @param token - Your API authorization token
 * @param text - The text to synthesize and play
 * @param options - Optional synthesis and playback parameters
 * @returns A promise that resolves to the API response
 */
export async function synthesizeAndPlay(
  baseUrl: string,
  token: string,
  text: string,
  options: TtsPlayOptions = {},
): Promise<TtsPlayResponse> {
  // Construct the query parameters
  const params = new URLSearchParams(
    Object.entries({
      voice: options.voice,
      lang: options.lang,
      play_now: options.playNow,
    })
      .filter(([_, value]) => value !== undefined && value !== "")
      .map(([key, value]) => [key, String(value)]),
  );

  const url = `${baseUrl.replace(/\/$/, "")}/api/tts/play?${params.toString()}`;

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "text/plain", // The body is raw text
      },
      body: text,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`SonicBoom API error (${response.status}): ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Failed to communicate with SonicBoom:", error);
    throw error;
  }
}

// --- Usage Example ---

const SONICBOOM_URL = "http://localhost:3000";
const MY_TOKEN = "sk-your-token-here";

synthesizeAndPlay(
  SONICBOOM_URL,
  MY_TOKEN,
  "Hello from the TypeScript client!",
  {
    voice: "F1",
    playNow: true,
  },
)
  .then((res) => console.log("Playing:", res.id))
  .catch((err) => console.error("Error:", err));
