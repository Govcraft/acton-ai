// Sample code for review demonstration
// This file has some intentional issues for the code review skill to find

fn process_data(data: Vec<i32>) -> i32 {
    let mut sum = 0;
    for i in 0..data.len() {
        sum = sum + data[i];
    }
    sum
}

fn get_user(id: i32) -> String {
    // TODO: implement database lookup
    format!("user_{}", id)
}

pub fn main_handler(input: String) -> Result<String, String> {
    let x = input.parse::<i32>();
    match x {
        Ok(n) => Ok(format!("Got number: {}", n)),
        Err(_) => Err("bad input".to_string()),
    }
}
