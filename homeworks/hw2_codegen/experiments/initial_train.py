import hydra
import train
from transformers import Seq2SeqTrainer, TrainerCallback


class StopAtCheckpoint(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= 2000:
            control.should_training_stop = True
            control.should_save = True
        return control


class InitialTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(StopAtCheckpoint())


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    train.LengthAwareTrainer = InitialTrainer
    train.main.__wrapped__(cfg)


if __name__ == "__main__":
    main()
