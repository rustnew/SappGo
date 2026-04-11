/// Available camera viewing modes for the MuJoCo visualizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraMode {
    /// Side view: azimuth 90°, slight downward angle.
    Side,
    /// Front view: azimuth 0°, slight downward angle.
    Front,
    /// Free orbit: user-controlled camera, no automatic positioning.
    Free,
}

impl CameraMode {
    /// Cycles to the next camera mode in order: Side → Front → Free → Side.
    #[inline]
    pub fn next(self) -> Self {
        match self {
            Self::Side  => Self::Front,
            Self::Front => Self::Free,
            Self::Free  => Self::Side,
        }
    }

    /// Human-readable name.
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Side  => "Side",
            Self::Front => "Front",
            Self::Free  => "Free",
        }
    }
}

impl std::fmt::Display for CameraMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
