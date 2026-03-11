use anyhow::Result;
use ogg::writing::PacketWriteEndInfo;
use opus::{Application, Channels, Encoder};

const OUTPUT_SAMPLE_RATE: u32 = 48000;
const FRAME_SIZE_48K: usize = 960; // 48000Hz 20ms

/// Decode OGG/Opus audio to raw PCM samples using symphonia
#[allow(dead_code)]
pub fn decode_ogg_to_samples(ogg_data: &[u8]) -> Result<(Vec<f32>, u32)> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    // Convert to owned data to avoid lifetime issues with Box<dyn MediaSource>
    let owned_data = ogg_data.to_vec();
    let mss = MediaSourceStream::new(Box::new(std::io::Cursor::new(owned_data)), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("ogg");
    hint.with_extension("opus");

    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();
    let decoder_opts = DecoderOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|e| anyhow::anyhow!("Failed to probe OGG file: {:?}", e))?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow::anyhow!("No audio track found"))?;

    let track_id = track.id;
    let codec_params = track.codec_params.clone();
    let sample_rate = codec_params.sample_rate.unwrap_or(48000);

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &decoder_opts)
        .map_err(|e| anyhow::anyhow!("Failed to create decoder: {:?}", e))?;

    let mut all_samples: Vec<f32> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(anyhow::anyhow!("Error reading packet: {:?}", e)),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("Decode error: {:?}", e);
                continue;
            }
        };

        let spec = *decoded.spec();
        let duration = decoded.capacity() as u64;

        if sample_buf.is_none() {
            sample_buf = Some(SampleBuffer::new(duration, spec));
        }

        if let Some(ref mut buf) = sample_buf {
            buf.copy_interleaved_ref(decoded);
            all_samples.extend_from_slice(buf.samples());
        }
    }

    Ok((all_samples, sample_rate))
}

/// Convert OGG/Opus data to WAV format using symphonia
#[allow(dead_code)]
pub fn convert_ogg_to_wav(ogg_data: &[u8], target_sample_rate: Option<u32>) -> Result<Vec<u8>> {
    let (samples, sample_rate) = decode_ogg_to_samples(ogg_data)?;

    // Resample if needed using rubato
    let final_samples = if let Some(target) = target_sample_rate {
        if target != sample_rate {
            resample_rubato(&samples, sample_rate, target)?
        } else {
            samples
        }
    } else {
        samples
    };

    encode_wav(&final_samples, target_sample_rate.unwrap_or(sample_rate))
}

/// Resample audio using linear interpolation (same as existing upsample_linear)
#[allow(dead_code)]
fn resample_rubato(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    Ok(upsample_linear(samples, from_rate, to_rate))
}

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
            AudioFormat::Opus => "audio/opus",
            AudioFormat::Wav => "audio/wav",
            AudioFormat::Mp3 => "audio/mpeg",
            AudioFormat::Flac => "audio/flac",
        }
    }

    #[allow(dead_code)]
    pub fn file_extension(&self) -> &'static str {
        match self {
            AudioFormat::Opus => "ogg",
            AudioFormat::Wav => "wav",
            AudioFormat::Mp3 => "mp3",
            AudioFormat::Flac => "flac",
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
    encode_ogg_opus(&opus_packets, sample_rate)
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
fn encode_ogg_opus(packets: &[Vec<u8>], input_sample_rate: u32) -> Result<Vec<u8>> {
    use std::io::Cursor;

    let mut out = Cursor::new(Vec::new());
    let serial = 0x12345678u32;
    let mut writer = ogg::writing::PacketWriter::new(&mut out);

    // ---- OpusHead header packet ----
    // https://wiki.xiph.org/OggOpus#ID_Header
    let mut opus_head = Vec::new();
    opus_head.extend_from_slice(b"OpusHead");
    opus_head.push(1); // version
    opus_head.push(1); // channel count (mono)
    opus_head.extend_from_slice(&0u16.to_le_bytes()); // pre-skip
    opus_head.extend_from_slice(&input_sample_rate.to_le_bytes()); // input sample rate
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

    // ---- Audio data packet ----
    // granule_pos = cumulative sample count (at 48000Hz)
    let frame_samples = FRAME_SIZE_48K as u64;
    let last_idx = packets.len().saturating_sub(1);
    for (i, packet) in packets.iter().enumerate() {
        let granule = frame_samples * (i as u64 + 1);
        let end_info = if i == last_idx {
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

/// Encode audio as MP3 format using shine-rs (pure Rust)
fn encode_mp3(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    use shine_rs::mp3_encoder::{encode_pcm_to_mp3, Mp3EncoderConfig, StereoMode};

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

    let mp3_data = encode_pcm_to_mp3(config, &pcm_i16).map_err(|e| anyhow::anyhow!("MP3 encoding failed: {:?}", e))?;
    Ok(mp3_data)
}

/// Encode audio as FLAC format using flacenc (pure Rust)
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
    
    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|e| anyhow::anyhow!("FLAC config error: {:?}", e))?;
    
    let source = flacenc::source::MemSource::from_samples(&pcm_i32, channels, bits_per_sample, sample_rate as usize);
    let flac_stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|e| anyhow::anyhow!("FLAC encode failed: {:?}", e))?;
    
    let mut sink = flacenc::bitsink::ByteSink::new();
    flac_stream.write(&mut sink).map_err(|e| anyhow::anyhow!("FLAC stream write failed: {:?}", e))?;
    
    Ok(sink.as_slice().to_vec())
}
