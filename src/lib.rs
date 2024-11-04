#![allow(clippy::borrow_deref_ref)]

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use bstr::ByteSlice;
use fancy_regex::Regex;
use std::collections::{HashMap, HashSet};
use ocaml::{List, Value, FromValue, ToValue, Runtime};
use lazy_static::lazy_static;

type Rank = u32;

const MAX_NUM_THREADS: usize = 128;

fn _byte_pair_merge(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<(usize, Rank)> {
    // This is a vector of (start, rank).
    // The rank is of the pair starting at position start.
    let mut parts = Vec::with_capacity(piece.len() + 1);

    // Note that we hash bytes when indexing into `ranks`, not token pairs. As long as we train BPE
    // the way we currently do, this is equivalent. An easy way to break this would be to decouple
    // merge priority from token index or to prevent specific token merges.
    let mut min_rank: (Rank, usize) = (Rank::MAX, usize::MAX);
    for i in 0..piece.len() - 1 {
        let rank = *ranks.get(&piece[i..i + 2]).unwrap_or(&Rank::MAX);
        if rank < min_rank.0 {
            min_rank = (rank, i);
        }
        parts.push((i, rank));
    }
    parts.push((piece.len() - 1, Rank::MAX));
    parts.push((piece.len(), Rank::MAX));

    let get_rank = {
        #[inline(always)]
        |parts: &Vec<(usize, Rank)>, i: usize| {
            if (i + 3) < parts.len() {
                // Similar to `piece[i..i + 2]` above. The +3 is because we haven't yet deleted
                // parts[i + 1], see comment in the main loop.
                *ranks
                    .get(&piece[parts[i].0..parts[i + 3].0])
                    .unwrap_or(&Rank::MAX)
            } else {
                Rank::MAX
            }
        }
    };

    // If you have n parts and m merges, this does O(mn) work.
    // We could do something with a heap and do O(m log n) work.
    // n is often very small so considerations like cache-locality outweigh the algorithmic
    // complexity downsides of the `parts` vector.
    while min_rank.0 != Rank::MAX {
        let i = min_rank.1;
        // Update parts[i] and parts[i - 1] before removing parts[i + 1], since
        // `parts.remove(i + 1)` will thrash the cache.
        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1);

        min_rank = (Rank::MAX, usize::MAX);
        for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
        }
    }
    parts
}

pub fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<Rank> {
    assert!(piece.len() > 1);
    _byte_pair_merge(&ranks, &piece)
        .windows(2)
        .map(|part| ranks[&piece[part[0].0..part[1].0]])
        .collect()
}

pub fn byte_pair_split<'a>(piece: &'a [u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<&'a [u8]> {
    assert!(piece.len() > 1);
    _byte_pair_merge(&ranks, &piece)
        .windows(2)
        .map(|part| &piece[part[0].0..part[1].0])
        .collect()
}

// CoreBPE struct definition
#[derive(Clone)]
struct CoreBPE {
    encoder: HashMap<Vec<u8>, Rank>,
    special_tokens_encoder: HashMap<String, Rank>,
    decoder: HashMap<Rank, Vec<u8>>,
    special_tokens_decoder: HashMap<Rank, Vec<u8>>,
    regex_tls: Vec<Regex>,
    special_regex_tls: Vec<Regex>,
    sorted_token_bytes: Vec<Vec<u8>>,
}

impl CoreBPE {
    fn new(
        encoder: HashMap<Vec<u8>, Rank>,
        special_tokens_encoder: HashMap<String, Rank>,
        pattern: &str,
    ) -> Result<Self, String> {
        let regex = Regex::new(pattern).map_err(|e| e.to_string())?;

        let special_regex = {
            let _parts = special_tokens_encoder
                .keys()
                .map(|s| fancy_regex::escape(s))
                .collect::<Vec<_>>();
            Regex::new(&_parts.join("|")).map_err(|e| e.to_string())?
        };

        let decoder: HashMap<Rank, Vec<u8>> =
            encoder.iter().map(|(k, v)| (*v, k.clone())).collect();

        if encoder.len() != decoder.len() {
            return Err("Encoder and decoder must be of equal length.".to_string());
        }

        let special_tokens_decoder: HashMap<Rank, Vec<u8>> = special_tokens_encoder
            .iter()
            .map(|(k, v)| (*v, k.as_bytes().to_vec()))
            .collect();

        let mut sorted_token_bytes: Vec<Vec<u8>> = encoder.keys().cloned().collect();
        sorted_token_bytes.sort();

        Ok(CoreBPE {
            encoder,
            special_tokens_encoder,
            decoder,
            special_tokens_decoder,
            regex_tls: vec![regex.clone(); MAX_NUM_THREADS],
            special_regex_tls: vec![special_regex.clone(); MAX_NUM_THREADS],
            sorted_token_bytes,
        })
    }

    fn _get_tl_regex(&self) -> &Regex {
        &self.regex_tls[hash_current_thread() % MAX_NUM_THREADS]
    }

    fn _get_tl_special_regex(&self) -> &Regex {
        &self.special_regex_tls[hash_current_thread() % MAX_NUM_THREADS]
    }

    fn _decode_native(&self, tokens: &[Rank]) -> Vec<u8> {
        let mut ret = Vec::with_capacity(tokens.len() * 2);
        for token in tokens {
            let token_bytes = self
                .decoder
                .get(token)
                .unwrap_or_else(|| &self.special_tokens_decoder[token]);
            ret.extend(token_bytes);
        }
        ret
    }

    fn _encode_ordinary_native(&self, text: &str) -> Vec<Rank> {
        let regex = self._get_tl_regex();
        let mut ret = vec![];
        for mat in regex.find_iter(text) {
            let piece = mat.unwrap().as_str().as_bytes();
            match self.encoder.get(piece) {
                Some(token) => ret.push(*token),
                None => ret.extend(&byte_pair_encode(piece, &self.encoder)),
            }
        }
        ret
    }

    fn _encode_native(&self, text: &str, allowed_special: &HashSet<&str>) -> (Vec<Rank>, usize) {
        let special_regex = self._get_tl_special_regex();
        let regex = self._get_tl_regex();
        let mut ret = vec![];

        let mut start = 0;
        let mut last_piece_token_len = 0;
        loop {
            let mut next_special;
            let mut start_find = start;
            loop {
                next_special = special_regex.find_from_pos(text, start_find).unwrap();
                match next_special {
                    Some(m) => {
                        if allowed_special.contains(&text[m.start()..m.end()]) {
                            break;
                        }
                        start_find = m.start() + 1;
                    }
                    None => break,
                }
            }
            let end = next_special.map_or(text.len(), |m| m.start());

            for mat in regex.find_iter(&text[start..end]) {
                let piece = mat.unwrap().as_str().as_bytes();
                if let Some(token) = self.encoder.get(piece) {
                    last_piece_token_len = 1;
                    ret.push(*token);
                    continue;
                }
                let tokens = byte_pair_encode(piece, &self.encoder);
                last_piece_token_len = tokens.len();
                ret.extend(&tokens);
            }

            match next_special {
                Some(m) => {
                    let piece = m.as_str();
                    let token = self.special_tokens_encoder[piece];
                    ret.push(token);
                    start = m.end();
                    last_piece_token_len = 0;
                }
                None => break,
            }
        }
        (ret, last_piece_token_len)
    }

    fn _increase_last_piece_token_len(
        &self,
        tokens: Vec<Rank>,
        mut last_piece_token_len: usize,
    ) -> (Vec<Rank>, usize) {
        let token_is_all_space = |token| {
            self.decoder
                .get(token)
                .map(|token_bytes| {
                    token_bytes
                        .iter()
                        .rev()
                        .all(|&b| [b' ', b'\n', b'\t'].contains(&b))
                })
                .unwrap_or(false)
        };
        if last_piece_token_len > 0
            && token_is_all_space(&tokens[tokens.len() - last_piece_token_len])
        {
            while (last_piece_token_len < tokens.len())
                && token_is_all_space(&tokens[tokens.len() - last_piece_token_len - 1])
            {
                last_piece_token_len += 1;
            }
        }
        debug_assert!(last_piece_token_len <= tokens.len());
        (tokens, last_piece_token_len)
    }

    fn _encode_unstable_native(
        &self,
        text: &str,
        allowed_special: &HashSet<&str>,
    ) -> (Vec<Rank>, HashSet<Vec<Rank>>) {
        let (tokens, last_piece_token_len) = self._encode_native(text, allowed_special);
        if last_piece_token_len == 0 {
            return (tokens, HashSet::new());
        }
        let (mut tokens, last_piece_token_len) =
            self._increase_last_piece_token_len(tokens, last_piece_token_len);

        let unstable_bytes = self._decode_native(&tokens[tokens.len() - last_piece_token_len..]);
        tokens.truncate(tokens.len() - last_piece_token_len);

        let mut completions = HashSet::new();
        if unstable_bytes.is_empty() {
            return (tokens, completions);
        }

        let mut point = self
            .sorted_token_bytes
            .partition_point(|x| x.as_slice() < unstable_bytes.as_slice());
        while point < self.sorted_token_bytes.len()
            && self.sorted_token_bytes[point].starts_with(&unstable_bytes)
        {
            completions.insert(vec![
                self.encoder[self.sorted_token_bytes[point].as_slice()],
            ]);
            point += 1;
        }

        for i in 1..unstable_bytes.len() {
            let prefix = &unstable_bytes[..i];
            let suffix = &unstable_bytes[i..];
            let mut point = self
                .sorted_token_bytes
                .partition_point(|x| x.as_slice() < suffix);
            while point < self.sorted_token_bytes.len()
                && self.sorted_token_bytes[point].starts_with(suffix)
            {
                let possibility = [prefix, self.sorted_token_bytes[point].as_slice()].concat();
                let encoded = match std::str::from_utf8(&possibility) {
                    Ok(s) => self._encode_ordinary_native(s),
                    Err(_) => byte_pair_encode(&possibility, &self.encoder),
                };
                let mut seq = Vec::new();
                let mut seq_len = 0;
                for token in encoded {
                    seq.push(token);
                    seq_len += self.decoder[&token].len();
                    if seq_len >= unstable_bytes.len() {
                        break;
                    }
                }
                completions.insert(seq);
                point += 1;
            }
        }

        if unstable_bytes.len() > 1 {
            let last_decoded = bstr::decode_last_utf8(unstable_bytes.as_slice());
            if unstable_bytes.len() - last_decoded.1 > 0
                && last_decoded.0.map_or(false, |c| c.is_whitespace())
            {
                let mut reencoded = byte_pair_encode(
                    &unstable_bytes[..unstable_bytes.len() - last_decoded.1],
                    &self.encoder,
                );
                reencoded.extend(byte_pair_encode(
                    &unstable_bytes[unstable_bytes.len() - last_decoded.1..],
                    &self.encoder,
                ));
                completions.insert(reencoded);
            }
        }

        (tokens, completions)
    }

    pub fn encode_ordinary(&self, text: &str) -> Vec<Rank> {
        // Directly call the native encoding function
        self._encode_ordinary_native(text)
    }

    pub fn encode(&self, text: &str, allowed_special: HashSet<&str>) -> Vec<Rank> {
        // Directly call the native encoding function with allowed special tokens
        self._encode_native(text, &allowed_special).0
    }

    pub fn encode_bytes(&self, bytes: &[u8]) -> Vec<Rank> {
        match std::str::from_utf8(bytes) {
            Ok(text) => self._encode_ordinary_native(text),
            Err(e) => {
                let text = unsafe { std::str::from_utf8_unchecked(&bytes[..e.valid_up_to()]) };
                let (tokens, last_piece_token_len) = self._encode_native(text, &HashSet::new());
                let (mut tokens, last_piece_token_len) =
                    self._increase_last_piece_token_len(tokens, last_piece_token_len);
                if !tokens.is_empty() && last_piece_token_len > 0 {
                    let mut unstable_bytes =
                        self._decode_native(&tokens[tokens.len() - last_piece_token_len..]);
                    unstable_bytes.extend_from_slice(&bytes[e.valid_up_to()..]);

                    tokens.truncate(tokens.len() - last_piece_token_len);
                    match self.encoder.get(&unstable_bytes) {
                        Some(token) => tokens.push(*token),
                        None => tokens.extend(&byte_pair_encode(&unstable_bytes, &self.encoder)),
                    }
                }
                tokens
            }
        }
    }

    pub fn encode_with_unstable(
        &self,
        text: &str,
        allowed_special: HashSet<&str>,
    ) -> (Vec<Rank>, Vec<Vec<Rank>>) {
        let (tokens, completions_set) = self._encode_unstable_native(text, &allowed_special);
        let completions: Vec<Vec<Rank>> = completions_set.into_iter().collect();
        (tokens, completions)
    }


    pub fn encode_single_token(&self, piece: &[u8]) -> Result<Rank, String> {
        if let Some(token) = self.encoder.get(piece).copied() {
            return Ok(token);
        }
        if let Ok(piece_str) = std::str::from_utf8(piece) {
            if let Some(token) = self.special_tokens_encoder.get(piece_str).copied() {
                return Ok(token);
            }
        }
        Err(format!("Token not found for piece: {:?}", piece))
    }

    pub fn encode_single_piece(&self, piece: &[u8]) -> Vec<Rank> {
        if let Some(token) = self.encoder.get(piece) {
            return vec![*token];
        }
        byte_pair_encode(piece, &self.encoder)
    }

    pub fn decode_bytes(&self, tokens: Vec<Rank>) -> Vec<u8> {
        self._decode_native(&tokens)
    }

    pub fn decode_single_token_bytes(&self, token: Rank) -> Result<Vec<u8>, String> {
        if let Some(bytes) = self.decoder.get(&token) {
            return Ok(bytes.clone());
        }
        if let Some(bytes) = self.special_tokens_decoder.get(&token) {
            return Ok(bytes.clone());
        }
        Err(format!("Token {} not found", token))
    }

    pub fn token_byte_values(&self) -> Vec<Vec<u8>> {
        self.sorted_token_bytes.clone()
    }



    

}

// Global storage for CoreBPE instances
lazy_static! {
    static ref CORE_BPE_STORE: Arc<Mutex<HashMap<usize, CoreBPE>>> = Arc::new(Mutex::new(HashMap::new()));
    static ref CORE_BPE_COUNTER: AtomicUsize = AtomicUsize::new(1);
}

// Helper function to get a CoreBPE instance by ID
fn get_core_bpe_instance(id: usize) -> Option<CoreBPE> {
    CORE_BPE_STORE.lock().unwrap().get(&id).cloned()
}

// Function to create a new CoreBPE instance and return its ID
#[ocaml::func]
#[ocaml::sig("Value -> Value -> string -> int")]
pub fn core_bpe_new(
    encoder: Value,
    special_tokens_encoder: Value,
    pattern: String,
) -> usize {
    let encoder_list: List<Value> = encoder.into();
    let encoder_vec: Vec<Value> = encoder_list.into_vec();
    let mut encoder_map: HashMap<Vec<u8>, Rank> = HashMap::new();

    for val in encoder_vec {
        let tuple: (Vec<u8>, Rank) = val.into();
        encoder_map.insert(tuple.0, tuple.1);
    }

    let special_tokens_list: List<Value> = special_tokens_encoder.into();
    let special_tokens_vec: Vec<Value> = special_tokens_list.into_vec();
    let mut special_tokens_map: HashMap<String, Rank> = HashMap::new();

    for val in special_tokens_vec {
        let tuple: (String, Rank) = val.into();
        special_tokens_map.insert(tuple.0, tuple.1);
    }

    let core_bpe = CoreBPE::new(encoder_map, special_tokens_map, &pattern).unwrap();
    let id = CORE_BPE_COUNTER.fetch_add(1, Ordering::SeqCst);
    CORE_BPE_STORE.lock().unwrap().insert(id, core_bpe);

    id // Return the ID to OCaml
}

// Function to encode text using CoreBPE by ID
#[ocaml::func]
pub fn core_bpe_encode_ordinary(core_bpe_id: usize, text: String) -> Value {
    let rt = Runtime::init(); // Initialize the OCaml runtime

    if let Some(bpe) = get_core_bpe_instance(core_bpe_id) {
        let result = bpe.encode_ordinary(&text);
        result.to_value(&rt) // Convert the result to OCaml value using the Runtime
    } else {
        (-1isize).to_value(&rt) // Return error as isize if ID is invalid
    }
}

// Function to decode tokens using CoreBPE by ID
#[ocaml::func]
pub fn core_bpe_decode_bytes(core_bpe_id: usize, tokens: Value) -> Value {
    let rt = Runtime::init(); // Initialize the OCaml runtime

    if let Some(bpe) = get_core_bpe_instance(core_bpe_id) {
        let tokens: Vec<Rank> = tokens.into();  // Convert OCaml Value to Vec<Rank>
        let result = bpe.decode_bytes(tokens);  // Decode the bytes using CoreBPE
        result.to_value(&rt)  // Convert the result (Vec<u8>) to an OCaml value
    } else {
        (-1isize).to_value(&rt)  // Return an error as an OCaml integer if ID is invalid
    }
}

fn hash_current_thread() -> usize {
    use std::num::NonZeroU64;
    use std::thread;

    struct FakeThreadId(NonZeroU64);

    const _: [u8; 8] = [0; std::mem::size_of::<thread::ThreadId>()];
    const _: [u8; 8] = [0; std::mem::size_of::<FakeThreadId>()];
    let x = unsafe {
        std::mem::transmute::<thread::ThreadId, FakeThreadId>(thread::current().id()).0
    };
    u64::from(x) as usize
}
