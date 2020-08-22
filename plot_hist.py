#  Copyright 2020 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import pickle
from matplotlib import pyplot as plt

with open("good_hopper_r", "rb") as handle:
    good_rs = pickle.load(handle)

with open("old_hopper_r", "rb") as handle:
    old_rs = pickle.load(handle)

plt.hist(good_rs, None, alpha=0.5, label='fine-tuned r')
plt.hist(old_rs, None, alpha=0.5, label='zero-shot r')
plt.legend(loc='upper right')
plt.show()