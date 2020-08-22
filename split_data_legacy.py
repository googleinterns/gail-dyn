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

import gzip, pickle, pickletools

all_trajs_split = {}

with open("./laika_23_softfloor_n200.pkl", "rb") as handle:
    saved_file = pickle.load(handle)

    n_trajs = len(saved_file)
    for traj_idx, traj_tuples in saved_file.items():
        traj_tuples_split = []
        for traj_tuple in traj_tuples:
            assert len(traj_tuple) == 116
            traj_tuple_split = [
                traj_tuple[:52],
                traj_tuple[52:64],
                traj_tuple[64:116]
            ]
            traj_tuples_split.append(traj_tuple_split)

        all_trajs_split[traj_idx] = traj_tuples_split

with open("./laika_23_softfloor_n200_split.pkl", "wb") as handle:
    # print(all_trajs)
    pickle.dump(all_trajs_split, handle, protocol=pickle.HIGHEST_PROTOCOL)
