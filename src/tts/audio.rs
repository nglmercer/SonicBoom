use anyhow::Result;
use ogg::writing::PacketWriteEndInfo;
use opus::{Application, Channels, Encoder};

const OUTPUT_SAMPLE_RATE: u32 = 48000;
const FRAME_SIZE_48K: usize = 960; // 48000Hz 20ms

pub fn encode_opus(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    // 48000Hz로 업샘플 (선형 보간)
    let resampled = upsample_linear(samples, sample_rate, OUTPUT_SAMPLE_RATE);

    let mut encoder =
        Encoder::new(OUTPUT_SAMPLE_RATE, Channels::Mono, Application::Audio)?;

    // f32 PCM → i16 PCM
    let pcm_i16: Vec<i16> = resampled
        .iter()
        .map(|s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect();

    // 패딩하여 FRAME_SIZE 배수 맞추기
    let pad_len = (FRAME_SIZE_48K - pcm_i16.len() % FRAME_SIZE_48K) % FRAME_SIZE_48K;
    let mut padded = pcm_i16;
    padded.extend(vec![0i16; pad_len]);

    // Opus 패킷 인코딩
    let mut opus_packets: Vec<Vec<u8>> = Vec::new();
    let mut buf = vec![0u8; 4096];
    for frame in padded.chunks(FRAME_SIZE_48K) {
        let len = encoder.encode(frame, &mut buf)?;
        opus_packets.push(buf[..len].to_vec());
    }

    // OGG 컨테이너로 패키징
    encode_ogg_opus(&opus_packets, sample_rate)
}

/// 선형 보간 업샘플
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

/// Opus 패킷들을 OGG 스트림으로 묶기
fn encode_ogg_opus(packets: &[Vec<u8>], input_sample_rate: u32) -> Result<Vec<u8>> {
    use std::io::Cursor;

    let mut out = Cursor::new(Vec::new());
    let serial = 0x12345678u32;
    let mut writer = ogg::writing::PacketWriter::new(&mut out);

    // ---- OpusHead 헤더 패킷 ----
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

    // ---- OpusTags 헤더 패킷 ----
    let mut opus_tags = Vec::new();
    opus_tags.extend_from_slice(b"OpusTags");
    let vendor = b"SonicBoom";
    opus_tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    opus_tags.extend_from_slice(vendor);
    opus_tags.extend_from_slice(&0u32.to_le_bytes()); // comment list length = 0
    writer.write_packet(opus_tags, serial, PacketWriteEndInfo::EndPage, 0)?;

    // ---- 오디오 데이터 패킷 ----
    // granule_pos = 누적 샘플 수 (48000Hz 기준)
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
