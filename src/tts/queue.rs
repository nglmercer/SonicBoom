use anyhow::Result;
use rodio::{Decoder, OutputStream, Sink};
use std::collections::VecDeque;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use tokio::sync::oneshot;

/// Represents an audio item in the queue
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AudioItem {
    /// Unique identifier for the audio item
    pub id: String,
    /// Path to the audio file
    pub path: PathBuf,
}

/// State for the audio queue system
pub struct AudioQueue {
    /// Queue of audio items waiting to be played
    queue: VecDeque<AudioItem>,
    /// Current playing item
    current: Option<AudioItem>,
    /// Whether playback is paused
    paused: bool,
    /// Volume level (0.0 to 1.0)
    volume: f32,
}

impl AudioQueue {
    /// Create a new audio queue
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            current: None,
            paused: false,
            volume: 1.0,
        }
    }

    /// Add an audio item to the queue
    pub fn enqueue(&mut self, item: AudioItem) {
        self.queue.push_back(item);
    }

    /// Remove and return the next item from the queue
    pub fn dequeue(&mut self) -> Option<AudioItem> {
        self.queue.pop_front()
    }

    /// Get the current item being played
    pub fn current(&self) -> Option<&AudioItem> {
        self.current.as_ref()
    }

    /// Set the current item
    pub fn set_current(&mut self, item: Option<AudioItem>) {
        self.current = item;
    }

    /// Get queue length
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Set paused state
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Set volume
    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume.clamp(0.0, 1.0);
    }

    /// Get volume
    pub fn volume(&self) -> f32 {
        self.volume
    }

    /// Clear the queue
    pub fn clear(&mut self) {
        self.queue.clear();
        self.current = None;
    }
}

impl Default for AudioQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Commands for the audio thread
#[derive(Debug)]
pub enum AudioCommand {
    /// Add to queue
    Enqueue { id: String, path: PathBuf },
    /// Play immediately (clears queue)
    PlayNow { id: String, path: PathBuf },
    /// Play next in queue
    PlayNext,
    /// Pause
    Pause,
    /// Resume
    Resume,
    /// Stop and clear queue
    Stop,
    /// Set volume
    SetVolume(f32),
    /// Get status (returns sender for response)
    GetStatus(oneshot::Sender<QueueStatus>),
}

/// Thread-safe audio manager that runs rodio in a separate thread
pub struct AudioManager {
    /// Command sender for the audio thread
    command_tx: tokio::sync::mpsc::Sender<AudioCommand>,
}

impl AudioManager {
    /// Create a new audio manager and start the audio thread
    pub fn new() -> Result<Self> {
        let (command_tx, command_rx) = tokio::sync::mpsc::channel(100);
        
        // Spawn the audio thread
        std::thread::spawn(move || {
            audio_thread(command_rx);
        });
        
        Ok(Self { command_tx })
    }

    /// Add an audio file to the queue
    pub async fn add_to_queue(&self, id: String, path: PathBuf) {
        let _ = self.command_tx.send(AudioCommand::Enqueue { id, path }).await;
    }

    /// Play a specific audio file immediately (clears queue)
    pub async fn play_now(&self, id: String, path: PathBuf) {
        let _ = self.command_tx.send(AudioCommand::PlayNow { id, path }).await;
    }

    /// Play the next item in the queue
    pub async fn play_next(&self) {
        let _ = self.command_tx.send(AudioCommand::PlayNext).await;
    }

    /// Pause playback
    pub async fn pause(&self) {
        let _ = self.command_tx.send(AudioCommand::Pause).await;
    }

    /// Resume playback
    pub async fn resume(&self) {
        let _ = self.command_tx.send(AudioCommand::Resume).await;
    }

    /// Stop playback
    pub async fn stop(&self) {
        let _ = self.command_tx.send(AudioCommand::Stop).await;
    }

    /// Set volume
    pub async fn set_volume(&self, volume: f32) {
        let _ = self.command_tx.send(AudioCommand::SetVolume(volume)).await;
    }

    /// Get queue status
    pub async fn status(&self) -> QueueStatus {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = self.command_tx.send(AudioCommand::GetStatus(tx)).await;
        rx.await.unwrap_or(QueueStatus {
            current: None,
            queue_length: 0,
            is_playing: false,
            is_paused: false,
            volume: 1.0,
        })
    }
}

/// The audio thread that handles playback
fn audio_thread(mut command_rx: tokio::sync::mpsc::Receiver<AudioCommand>) {
    // Initialize rodio
    let (_stream, stream_handle) = match OutputStream::try_default() {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("Failed to initialize audio output: {}", e);
            return;
        }
    };

    let mut queue = AudioQueue::new();
    let mut sink: Option<Sink> = None;

    // Runtime for async operations
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    loop {
        // Check if current playback finished
        if let Some(s) = &sink {
            if s.empty() && !s.is_paused() {
                // Playback finished
                let current_id = queue.current.as_ref().map(|c| c.id.clone());
                if let Some(_id) = current_id {
                    // Playback ended, could signal here if needed
                }
                queue.set_current(None);
                sink = None;
                
                // Try to play next
                if let Some(item) = queue.dequeue() {
                    if let Ok(file) = File::open(&item.path) {
                        let reader = BufReader::new(file);
                        if let Ok(source) = Decoder::new(reader) {
                            if let Ok(new_sink) = Sink::try_new(&stream_handle) {
                                new_sink.set_volume(queue.volume());
                                new_sink.append(source);
                                sink = Some(new_sink);
                                queue.set_current(Some(item));
                            }
                        }
                    }
                }
            }
        }

        // Use blocking recv with timeout
        match rt.block_on(async {
            tokio::time::timeout(std::time::Duration::from_millis(100), command_rx.recv()).await
        }) {
            Ok(Some(cmd)) => {
                match cmd {
                    AudioCommand::Enqueue { id, path } => {
                        queue.enqueue(AudioItem { id, path });
                    }
                    AudioCommand::PlayNow { id, path } => {
                        queue.clear();
                        if let Some(s) = sink.take() {
                            s.stop();
                        }
                        
                        if path.exists() {
                            if let Ok(file) = File::open(&path) {
                                let reader = BufReader::new(file);
                                if let Ok(source) = Decoder::new(reader) {
                                    if let Ok(new_sink) = Sink::try_new(&stream_handle) {
                                        new_sink.set_volume(queue.volume());
                                        new_sink.append(source);
                                        sink = Some(new_sink);
                                        queue.set_current(Some(AudioItem { id, path }));
                                    }
                                }
                            }
                        }
                    }
                    AudioCommand::PlayNext => {
                        if let Some(s) = sink.take() {
                            s.stop();
                        }
                        
                        if let Some(item) = queue.dequeue() {
                            if item.path.exists() {
                                if let Ok(file) = File::open(&item.path) {
                                    let reader = BufReader::new(file);
                                    if let Ok(source) = Decoder::new(reader) {
                                        if let Ok(new_sink) = Sink::try_new(&stream_handle) {
                                            new_sink.set_volume(queue.volume());
                                            new_sink.append(source);
                                            sink = Some(new_sink);
                                            queue.set_current(Some(item));
                                        }
                                    }
                                }
                            }
                        } else {
                            queue.set_current(None);
                        }
                    }
                    AudioCommand::Pause => {
                        if let Some(s) = &sink {
                            s.pause();
                        }
                        queue.set_paused(true);
                    }
                    AudioCommand::Resume => {
                        if let Some(s) = &sink {
                            s.play();
                        }
                        queue.set_paused(false);
                    }
                    AudioCommand::Stop => {
                        if let Some(s) = sink.take() {
                            s.stop();
                        }
                        queue.clear();
                        sink = None;
                    }
                    AudioCommand::SetVolume(volume) => {
                        let vol = volume.clamp(0.0, 1.0);
                        if let Some(s) = &sink {
                            s.set_volume(vol);
                        }
                        queue.set_volume(vol);
                    }
                    AudioCommand::GetStatus(tx) => {
                        let is_playing = sink.as_ref().map(|s| !s.is_paused() && !s.empty()).unwrap_or(false);
                        let is_paused = sink.as_ref().map(|s| s.is_paused()).unwrap_or(false);
                        
                        let _ = tx.send(QueueStatus {
                            current: queue.current().cloned(),
                            queue_length: queue.len(),
                            is_playing,
                            is_paused,
                            volume: queue.volume(),
                        });
                    }
                }
            }
            Ok(None) => break, // Channel closed
            Err(_) => {
                // Timeout, continue loop
            }
        }
    }
}

/// Status information about the queue
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QueueStatus {
    pub current: Option<AudioItem>,
    pub queue_length: usize,
    pub is_playing: bool,
    pub is_paused: bool,
    pub volume: f32,
}
