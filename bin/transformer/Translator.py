''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Translator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):
        

        super(Translator, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs', 
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map', 
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))


    def _model_decode(self, trg_seq, enc_output, src_seq, device):
        def get_trg_pos(trg_seq):
            n_row, len_seq = trg_seq.size()
            trg_pos = torch.arange(1, len_seq + 1, dtype=torch.long, device=device)
            trg_pos = trg_pos.unsqueeze(0).repeat(n_row, 1)
            return trg_pos
        trg_pos = get_trg_pos(trg_seq)
        dec_output, *_ = self.model.decoder(trg_seq, trg_pos, src_seq, enc_output)
        return F.log_softmax(self.model.tgt_word_prj(dec_output), dim=-1)


    def _get_init_state(self, src_seq, src_pos, device):
        enc_output, *_ = self.model.encoder(src_seq, src_pos)
        batch_size, len_seq = src_seq.size()
        dec_output = self._model_decode(self.init_seq.expand(batch_size, -1), 
                                        enc_output, src_seq, device)
        
        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(self.beam_size)
        gen_seq = self.blank_seqs.repeat(batch_size, 1)
        gen_seq[:, 1] = best_k_idx.view(-1)
        vec_size = enc_output.size(2)
        enc_output = enc_output.repeat(1, self.beam_size, 1).view(-1, len_seq, vec_size)
        scores = best_k_probs.view(-1, 1).repeat(1, self.beam_size)
        src_seq = src_seq.repeat(1, self.beam_size).view(batch_size * self.beam_size, len_seq)
        return src_seq, enc_output, gen_seq, scores


    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step, device):
        
        batch_size = gen_seq.size(0) // self.beam_size
        
        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(self.beam_size)

        # Include the previous scores.
        scores = best_k2_probs + scores

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(scores.size(0) // self.beam_size, -1).topk(self.beam_size)
        scores = scores.view(-1, 1).repeat(1, self.beam_size)
 
        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // self.beam_size, best_k_idx_in_k2 % self.beam_size
        best_k_r_idxs += torch.tensor([batch_idx * self.beam_size for batch_idx in range(batch_size)], device=device).view(-1, 1).repeat(1, self.beam_size)
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step].view(-1, step)
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx.view(-1)

        return gen_seq, scores


    def translate_batch(self, batch, device):
        src_seq, src_pos = batch[0], batch[1]

        with torch.no_grad():
            src_seq = src_seq.to(device)
            src_pos = src_pos.to(device)
            src_seq, enc_output, gen_seq, scores = self._get_init_state(src_seq, src_pos, device)
            
            ans_idx = 0   # default
            batch_beam_size = src_seq.size(0)
            len_map = self.len_map.repeat(1, batch_beam_size).view(batch_beam_size, self.max_seq_len)
            for step in range(2, self.max_seq_len):    # decode up to max length
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_seq, device)
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step, device)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == self.trg_eos_idx
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = len_map.masked_fill(~eos_locs, self.max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == batch_beam_size:
                    ans_idxs = torch.tensor([batch_idx * self.beam_size for batch_idx in range(batch_beam_size //self.beam_size)], device=device)
                    break
        return gen_seq[ans_idxs].tolist()
