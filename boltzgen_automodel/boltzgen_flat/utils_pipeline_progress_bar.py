"""
Custom PyTorch Lightning progress bar callback that displays pipeline step information.
"""
import os

from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm import tqdm


class PipelineProgressBar(TQDMProgressBar):
    """Custom progress bar that shows the current pipeline step."""
    
    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        """Initialize the pipeline progress bar.
        
        Parameters
        ----------
        refresh_rate : int
            Refresh rate for the progress bar
        process_position : int
            Position of the process for multi-processing
        """
        super().__init__(refresh_rate=refresh_rate, process_position=process_position)
        self._original_tqdm = None
        self._patch_tqdm()
    
    def _get_pipeline_info(self) -> str:
        """Get pipeline step information from environment variables."""
        pipeline_step = os.environ.get("BOLTZGEN_PIPELINE_STEP", "")
        pipeline_progress = os.environ.get("BOLTZGEN_PIPELINE_PROGRESS", "")
        
        if pipeline_step:
            if pipeline_progress:
                return f"[{pipeline_progress}] {pipeline_step}"
            else:
                return f"[Pipeline] {pipeline_step}"
        return ""
    
    def _update_bar_description(self, bar):
        """Helper method to update a progress bar description with pipeline info."""
        if bar is not None:
            pipeline_info = self._get_pipeline_info()
            if pipeline_info:
                # Update the description to include pipeline step info
                current_desc = getattr(bar, 'desc', '') or ''
                if current_desc:
                    new_desc = f"{pipeline_info} - {current_desc}"
                    bar.set_description(new_desc)
                else:
                    bar.set_description(pipeline_info)
        return bar
    
    def init_predict_tqdm(self) -> None:
        """Initialize the prediction progress bar."""
        bar = super().init_predict_tqdm()
        return self._update_bar_description(bar)
    
    def init_train_tqdm(self) -> None:
        """Initialize the training progress bar."""
        bar = super().init_train_tqdm()
        return self._update_bar_description(bar)
    
    def init_validation_tqdm(self) -> None:
        """Initialize the validation progress bar."""
        bar = super().init_validation_tqdm()
        return self._update_bar_description(bar)
    
    def init_test_tqdm(self) -> None:
        """Initialize the test progress bar."""
        bar = super().init_test_tqdm()
        return self._update_bar_description(bar)

    def init_sanity_tqdm(self) -> None:
        """Initialize the sanity check progress bar."""
        bar = super().init_sanity_tqdm()
        return self._update_bar_description(bar)
    
    def on_predict_start(self, trainer, pl_module):
        """Called when prediction starts - update all progress bars."""
        super().on_predict_start(trainer, pl_module)
        self._update_all_progress_bars()
    
    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Called before each prediction batch - update progress bars."""
        super().on_predict_batch_start(trainer, pl_module, batch, batch_idx)
        self._update_all_progress_bars()
    
    def _update_all_progress_bars(self):
        """Update all existing progress bars with pipeline info."""
        pipeline_info = self._get_pipeline_info()
        if not pipeline_info:
            return
            
        # Try to find and update all progress bars
        for attr_name in dir(self):
            if ('progress' in attr_name.lower() or 
                'tqdm' in attr_name.lower() or 
                attr_name.endswith('_bar')):
                try:
                    bar = getattr(self, attr_name)
                    if (bar is not None and 
                        hasattr(bar, 'set_description') and 
                        hasattr(bar, 'desc')):
                        current_desc = getattr(bar, 'desc', '') or ''
                        if pipeline_info not in current_desc:
                            if current_desc:
                                new_desc = f"{pipeline_info} - {current_desc}"
                            else:
                                new_desc = pipeline_info
                            bar.set_description(new_desc)
                except (AttributeError, TypeError):
                    pass
    
    def print(self, *args, **kwargs):
        """Override print method to intercept and modify all progress updates."""
        # Check if this is updating a progress bar description and modify it
        if args:
            # Try to update any active progress bars before printing
            self._update_all_progress_bars()
        
        return super().print(*args, **kwargs)
    
    def get_metrics(self, trainer, pl_module):
        """Override get_metrics to update progress bars on each metric update."""
        metrics = super().get_metrics(trainer, pl_module)
        self._update_all_progress_bars()
        return metrics
    
    def _patch_tqdm(self):
        """Patch tqdm to automatically add pipeline info to all progress bars."""
        if self._original_tqdm is not None:
            return  # Already patched
            
        # Store the original tqdm class
        self._original_tqdm = tqdm
        
        # Create a wrapper that adds pipeline info
        def patched_tqdm(*args, **kwargs):
            # Create the original tqdm instance
            instance = self._original_tqdm(*args, **kwargs)
            
            # Update its description with pipeline info
            pipeline_info = self._get_pipeline_info()
            if pipeline_info:
                current_desc = kwargs.get('desc', '') or getattr(instance, 'desc', '') or ''
                if pipeline_info not in current_desc:
                    if current_desc and current_desc.strip():
                        # Remove any trailing colons from current description to avoid double colons
                        current_desc = current_desc.rstrip(':').strip()
                        new_desc = f"{pipeline_info} - {current_desc}"
                    else:
                        new_desc = pipeline_info
                    instance.set_description(new_desc)
            
            return instance
        
        # Monkey patch tqdm globally during our callback's lifetime
        import tqdm as tqdm_module
        tqdm_module.tqdm = patched_tqdm
        
    def _unpatch_tqdm(self):
        """Restore original tqdm."""
        if self._original_tqdm is not None:
            import tqdm as tqdm_module
            tqdm_module.tqdm = self._original_tqdm
            self._original_tqdm = None
    
    def teardown(self, trainer, pl_module, stage):
        """Called when the callback is being torn down."""
        super().teardown(trainer, pl_module, stage)
        self._unpatch_tqdm()
