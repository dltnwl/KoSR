/* MIT License

Copyright (c) 2018 Suji Lee and Seokjin Han

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. */

#ifndef CUSTOM_JAMO_SEARCH_H_
#define CUSTOM_JAMO_SEARCH_H_

#include "ctc_beam_search.h"
#include "ctc_loss_util.h"

namespace tf = tensorflow;

const int MOVE[4][54] = {
//    ㄱ ㄲ    ㄴ       ㄷ ㄸ ㄹ                      ㅁ ㅂ ㅃ    ㅅ ㅆ ㅇ ㅈ ㅉ ㅊ ㅋ ㅌ ㅍ ㅎ
  {-1, 1, 1,-1, 1,-1,-1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0},
//                                                                                              ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅗ ㅘ ㅙ ㅚ ㅛ ㅜ ㅝ ㅞ ㅟ ㅠ ㅡ ㅢ ㅣ
  {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,-1, 1},
//  ∅ ㄱ ㄲ ㄳ ㄴ ㄵ ㄶ ㄷ    ㄹ ㄺ ㄻ ㄼ ㄽ ㄾ ㄿ ㅀ ㅁ ㅂ    ㅄ ㅅ ㅆ ㅇ ㅈ    ㅊ ㅋ ㅌ ㅍ ㅎ
  { 3, 3, 3, 3, 3, 3, 3, 3,-1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,-1, 3, 3, 3, 3, 3,-1, 3, 3, 3, 3, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 2},
//    ㄱ ㄲ    ㄴ       ㄷ ㄸ ㄹ                      ㅁ ㅂ ㅃ    ㅅ ㅆ ㅇ ㅈ ㅉ ㅊ ㅋ ㅌ ㅍ ㅎ
  {-1, 1, 1,-1, 1,-1,-1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 3}};
//  ∅ ㄱ ㄲ ㄳ ㄴ ㄵ ㄶ ㄷ ㄸ ㄹ ㄺ ㄻ ㄼ ㄽ ㄾ ㄿ ㅀ ㅁ ㅂ ㅃ ㅄ ㅅ ㅆ ㅇ ㅈ ㅉ ㅊ ㅋ ㅌ ㅍ ㅎ ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅗ ㅘ ㅙ ㅚ ㅛ ㅜ ㅝ ㅞ ㅟ ㅠ ㅡ ㅢ ㅣ  _  blank

struct JamoState {
  int phase;
};

class JamoScorer : public tf::ctc::BaseBeamScorer<JamoState> {
 public:
  void InitializeState(JamoState* root) const override {
    root->phase = 0;
  }
  void ExpandState(const JamoState& from_state, int from_label,
                   JamoState* to_state, int to_label) const override {
    to_state->phase = MOVE[from_state.phase][to_label];
  }
  float GetStateExpansionScore(const JamoState& state,
                               float previous_score) const override {
    return state.phase < 0 ? tf::ctc::kLogZero : previous_score;
  }
};

#endif  // CUSTOM_JAMO_SEARCH_H_
