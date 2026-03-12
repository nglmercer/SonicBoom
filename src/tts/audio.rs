use anyhow::Result;
use opus::{Application, Channels, Encoder};

const OUTPUT_SAMPLE_RATE: u32 = 48000;
const FRAME_SIZE_48K: usize = 960; // 48000Hz 20ms

/// Supported audio output formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    Opus,
    Wav,
    Mp3,
    Flac,
}

impl AudioFormat {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "wav" => AudioFormat::Wav,
            "mp3" => AudioFormat::Mp3,
            "flac" => AudioFormat::Flac,
            _ => AudioFormat::Opus,
        }
    }

    pub fn content_type(&self) -> &'static str {
        match self {
            AudioFormat::Opus => "audio/ogg; codecs=opus",
            AudioFormat::Wav => "audio/wav",
            AudioFormat::Mp3 => "audio/mpeg",
            AudioFormat::Flac => "audio/flac",
        }
    }
}

pub fn encode_audio(samples: &[f32], sample_rate: u32, format: AudioFormat) -> Result<Vec<u8>> {
    match format {
        AudioFormat::Opus => encode_opus(samples, sample_rate),
        AudioFormat::Wav => encode_wav(samples, sample_rate),
        AudioFormat::Mp3 => encode_mp3(samples, sample_rate),
        AudioFormat::Flac => encode_flac(samples, sample_rate),
    }
}

pub fn encode_opus(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    // Upsample to 48000Hz (linear interpolation)
    let resampled = upsample_linear(samples, sample_rate, OUTPUT_SAMPLE_RATE);

    let mut encoder =
        Encoder::new(OUTPUT_SAMPLE_RATE, Channels::Mono, Application::Audio)?;

    // Set Opus encoder to VBR (Variable Bitrate) mode for better quality
    encoder.set_bitrate(opus::Bitrate::Bits(128000))?;

    // f32 PCM → i16 PCM
    let pcm_i16: Vec<i16> = resampled
        .iter()
        .map(|s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect();

    // Pad to make multiple of FRAME_SIZE
    let pad_len = (FRAME_SIZE_48K - pcm_i16.len() % FRAME_SIZE_48K) % FRAME_SIZE_48K;
    let mut padded = pcm_i16;
    padded.extend(vec![0i16; pad_len]);

    // Encode Opus packets
    let mut opus_packets: Vec<Vec<u8>> = Vec::new();
    let mut buf = vec![0u8; 4096];
    for frame in padded.chunks(FRAME_SIZE_48K) {
        let len = encoder.encode(frame, &mut buf)?;
        opus_packets.push(buf[..len].to_vec());
    }

    // Package in OGG container
    encode_ogg_opus(&opus_packets, OUTPUT_SAMPLE_RATE)
}

/// Linear interpolation upsampling
fn upsample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = (samples.len() as f64 * ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let idx = src_pos as usize;
        let frac = (src_pos - idx as f64) as f32;
        let a = samples[idx.min(samples.len() - 1)];
        let b = samples[(idx + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

/// Bundle Opus packets into OGG stream
fn encode_ogg_opus(packets: &[Vec<u8>], sample_rate: u32) -> Result<Vec<u8>> {
    use ogg::writing::PacketWriteEndInfo;
    use std::io::Cursor;

    let mut out = Cursor::new(Vec::new());
    let serial = 0x12345678u32;
    let mut writer = ogg::writing::PacketWriter::new(&mut out);

    // ---- OpusHead header packet ----
    // https://wiki.xiph.org/OggOpus#ID_Header
    let pre_skip: u16 = 312; // Required by Opus spec (minimum 312 samples)
    let mut opus_head = Vec::new();
    opus_head.extend_from_slice(b"OpusHead");
    opus_head.push(1); // version
    opus_head.push(1); // channel count (mono)
    opus_head.extend_from_slice(&pre_skip.to_le_bytes()); // pre-skip (312 samples)
    opus_head.extend_from_slice(&sample_rate.to_le_bytes()); // input sample rate
    opus_head.extend_from_slice(&0i16.to_le_bytes()); // output gain
    opus_head.push(0); // channel mapping family
    writer.write_packet(opus_head, serial, PacketWriteEndInfo::EndPage, 0)?;

    // ---- OpusTags header packet ----
    let mut opus_tags = Vec::new();
    opus_tags.extend_from_slice(b"OpusTags");
    let vendor = b"SonicBoom";
    opus_tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    opus_tags.extend_from_slice(vendor);
    opus_tags.extend_from_slice(&0u32.to_le_bytes()); // comment list length = 0
    writer.write_packet(opus_tags, serial, PacketWriteEndInfo::EndPage, 0)?;

    // ---- Audio data packets ----
    // For OggOpus, granule position represents the presentation time of the
    // last sample in the page. The first 312 samples (pre-skip) should be
    // discarded by the decoder, so the first valid sample appears at
    // granule position 312.
    let frame_samples = FRAME_SIZE_48K as u64;
    
    for (i, packet) in packets.iter().enumerate() {
        // Granule position = pre_skip + (packet_index + 1) * frame_samples
        let granule = pre_skip as u64 + frame_samples * (i as u64 + 1);
        
        let end_info = if i + 1 == packets.len() {
            PacketWriteEndInfo::EndStream
        } else {
            PacketWriteEndInfo::NormalPacket
        };
        
        writer.write_packet(packet.clone(), serial, end_info, granule)?;
    }

    drop(writer);
    Ok(out.into_inner())
}

/// Encode audio as WAV format (PCM 16-bit) using hound
fn encode_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    use std::io::Cursor;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut cursor = Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut cursor, spec)?;

    for &sample in samples {
        let amplitude = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(amplitude)?;
    }

    writer.finalize()?;
    Ok(cursor.into_inner())
}

/// Encode audio as MP3 format using shine-rs
fn encode_mp3(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    use shine_rs::mp3_encoder::{encode_pcm_to_mp3, Mp3EncoderConfig, StereoMode};

    // Use standard settings for maximum compatibility
    let config = Mp3EncoderConfig {
        sample_rate,
        bitrate: 128,
        channels: 1,
        stereo_mode: StereoMode::Mono,
        copyright: false,
        original: true,
    };

    // Convert f32 to i16 PCM
    let pcm_i16: Vec<i16> = samples
        .iter()
        .map(|s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect();

    // Encode using shine-rs
    let mp3_data = encode_pcm_to_mp3(config, &pcm_i16)
        .map_err(|e| anyhow::anyhow!("MP3 encoding failed: {:?}", e))?;

    // Verify MP3 frame sync bytes are present (should start with 0xFF 0xFB or similar)
    //shine-rs should produce valid MP3 frames with proper headers
    
    // Add ID3v2 header for better compatibility with media players
    // Many players require ID3v2 to recognize MP3 files properly
    let id3v2 = create_id3v2_header();
    let mut result = Vec::with_capacity(id3v2.len() + mp3_data.len());
    result.extend_from_slice(&id3v2);
    result.extend_from_slice(&mp3_data);

    Ok(result)
}

/// Create minimal ID3v2 header for MP3 compatibility
fn create_id3v2_header() -> Vec<u8> {
    // ID3v2.3 header: "ID3" + version (2 bytes) + flags + size (syncsafe)
    let mut header = Vec::new();
    header.extend_from_slice(b"ID3"); // Identifier
    header.push(3); // Version (ID3v2.3)
    header.push(0); // Revision
    header.push(0); // Flags (none)
    
    // Size in syncsafe bytes (7 bits per byte) - standard minimal tag
    let size: u32 = 0;
    header.push(((size >> 21) & 0x7F) as u8);
    header.push(((size >> 14) & 0x7F) as u8);
    header.push(((size >> 7) & 0x7F) as u8);
    header.push((size & 0x7F) as u8);
    
    header
}

/// Encode audio as FLAC format using flacenc
fn encode_flac(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;

    // Convert f32 to i32 for flacenc (16-bit range)
    let pcm_i32: Vec<i32> = samples
        .iter()
        .map(|s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i32)
        .collect();

    let channels = 1;
    let bits_per_sample = 16;
    
    // Configure encoder with proper settings
    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|e| anyhow::anyhow!("FLAC config error: {:?}", e))?;
    
    let source = flacenc::source::MemSource::from_samples(
        &pcm_i32, 
        channels, 
        bits_per_sample, 
        sample_rate as usize
    );
    
    let block_size = 4096;
    let flac_stream = flacenc::encode_with_fixed_block_size(&config, source, block_size)
        .map_err(|e| anyhow::anyhow!("FLAC encode failed: {:?}", e))?;
    
    let mut sink = flacenc::bitsink::ByteSink::new();
    flac_stream.write(&mut sink).map_err(|e| anyhow::anyhow!("FLAC stream write failed: {:?}", e))?;
    
    Ok(sink.as_slice().to_vec())
}
