from typing import Optional

from location_manager.seed_location_manager import SeedLocationManager
from univariate_feature_selection.univariate_feature_selector_descriptor import  \
    DEFAULT_CATEGORICAL_FS_DESCRIPTOR, CompositeManyFeatureSelectorDescriptor, \
    DEFAULT_SURVIVAL_FS_DESCRIPTOR, ManyFeatureSelectorClassDescriptor, FeatureSelectorMOUnionDescriptor


class FsLocationManager(SeedLocationManager):

    def _fs_string(
            self,
            categorical_fs_descriptor: ManyFeatureSelectorClassDescriptor = DEFAULT_CATEGORICAL_FS_DESCRIPTOR
    ) -> Optional[str]:
        fs_so_descriptor = CompositeManyFeatureSelectorDescriptor(
            categorical_selector = categorical_fs_descriptor,
            survival_selector= DEFAULT_SURVIVAL_FS_DESCRIPTOR
        )
        fs_mo_descriptor = FeatureSelectorMOUnionDescriptor(feature_selector_so=fs_so_descriptor)
        return "_" + fs_mo_descriptor.nick()
