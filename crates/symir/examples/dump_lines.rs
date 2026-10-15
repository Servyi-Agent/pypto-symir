// Test expectations read better inline; the servyi `inline_literal` policy
// sanctions silencing sites with an explicit allow.
#![allow(inline_literal)]
fn main() {
    let path = std::env::args().nth(1).unwrap_or_default();
    let src = std::fs::read_to_string(&path).unwrap_or_default();
    match symir::parse::parse_program(&src) {
        Ok(p) => {
            for f in &p.functions {
                println!(
                    "fn {} params {:?} stmts {}",
                    f.name,
                    f.params.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
                    f.body.len()
                );
                for st in &f.body {
                    println!("  stmt: {st:?}");
                }
            }
            match symir::interp::interpret(&p, None) {
                Ok(dag) => {
                    for n in &dag.nodes {
                        println!("node {} {:?}", n.id, n.kind);
                    }
                    println!("edges: {}", dag.edges.len());
                }
                Err(e) => println!("INTERP ERR {e:?}"),
            }
        }
        Err(e) => println!("ERR {e:?}"),
    }
}
