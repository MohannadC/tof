use std::io::{BufRead, BufReader, BufWriter, Write, Seek, SeekFrom};
use std::fs::{OpenOptions, File};
use std::collections::HashMap;
use std::time::Instant;
use std::env;

fn main() -> Result<(), std::io::Error> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        panic!("Specify an input file path in arguments:
             -> ./data_handler.exe file.txt");
    }
    let path = "./data/".to_owned()+&args[1];
    match File::open(&path) {
        Err(_) => {
            println!("Couldn't open a file");
        },
        Ok(_) => {
            println!("File was opened successfully");
            let reader = BufReader::new(File::open(&path)?);

            let now = Instant::now();
            let mut data: HashMap<i64, f64> = HashMap::new();
            let mut tmp: i64 = 0;
            for line in reader.lines() {
                let line = line?;
                let collection = line.split_whitespace().collect::<Vec<&str>>();
                let number: i64 = collection[1].parse::<f64>().unwrap().round() as i64;
                let time: f64 = collection[0].parse::<f64>().unwrap();
                
                if number != 0 && number != tmp && !data.contains_key(&number){
                    for i in 1..(number-tmp+1) {
                        data.insert(tmp+i, time);
                    }
                    tmp = number;
                }
            }

            let mut file = OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap();
            let _ = File::set_len(&file, 0);
            let _ = file.seek(SeekFrom::Start(0));
            let mut buf = BufWriter::new(file);
            for i in 1..tmp+1 {
                if let Err(e) = writeln!(buf, "{}   {}", data[&i], i) {
                    eprintln!("Couldn't write to file: {}", e);
                }
            }

            println!("Time elapsed: {:.2?}", now.elapsed());
            println!("Process finished with exit code 0");
        }
    }
    Ok(())
}