use anyhow::*;
use fs_extra::copy_items;
use fs_extra::dir::CopyOptions;
use std::env;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=res/*");

    let mut copy_options = CopyOptions::new();
    copy_options.overwrite = true;
    let paths_to_copy = vec!["res/"];

    let out_path = std::path::Path::new("target")
        .join(env::var("PROFILE")?);
    copy_items(&paths_to_copy, &out_path, &copy_options)?;
    println!("Resources copied to {}", out_path.display());
    Ok(())
}