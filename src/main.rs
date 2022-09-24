mod render_system;

use render_system::run;

fn main() {
    pollster::block_on(run());
}
