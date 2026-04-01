use scry_mermaid::prelude::*;

fn main() {
    let diagrams: Vec<(&str, &str, MermaidTheme)> = vec![
        (
            "ci_pipeline",
            r#"graph TD
    A[Push to Main] --> B{Tests Pass?}
    B -->|Yes| C[Build Docker Image]
    B -->|No| D[Notify Developer]
    D --> E[Fix Code]
    E --> A
    C --> F{Deploy to Staging?}
    F -->|Yes| G([Staging Deploy])
    G --> H{Smoke Tests Pass?}
    H -->|Yes| I([Production Deploy])
    H -->|No| D
    F -->|No| J((End))"#,
            MermaidTheme::dark(),
        ),
        (
            "microservices",
            r#"flowchart LR
    A([API Gateway]) --> B[Auth Service]
    A --> C[User Service]
    A --> D[Order Service]
    B --> E[(Database)]
    C --> E
    D --> F[(Order DB)]
    D --> G[Payment Service]
    G --> H[Stripe API]"#,
            MermaidTheme::dark(),
        ),
        (
            "git_flow",
            r#"graph TD
    A[Feature Branch] --> B{Code Review}
    B -->|Approved| C[Merge to Dev]
    B -->|Changes Requested| D[Revise]
    D --> B
    C --> E{QA Pass?}
    E -->|Yes| F[Merge to Main]
    E -->|No| G[Bug Fix]
    G --> C
    F --> H([Release Tag])
    H --> I((Deploy))"#,
            MermaidTheme::light(),
        ),
        (
            "data_pipeline",
            r#"flowchart LR
    A[[Ingestion]] --> B[Validate]
    B --> C{Schema OK?}
    C -->|Yes| D[Transform]
    C -->|No| E[Dead Letter Queue]
    D --> F[[Enrichment]]
    F --> G[Aggregate]
    G --> H[(Data Warehouse)]
    H --> I([Dashboard])
    H --> J([ML Pipeline])"#,
            MermaidTheme::dark(),
        ),
        (
            "auth_flow",
            r#"graph TD
    A((Start)) --> B[User Enters Credentials]
    B --> C{Valid?}
    C -->|No| D[Show Error]
    D --> B
    C -->|Yes| E{MFA Enabled?}
    E -->|No| F([Grant Access])
    E -->|Yes| G[Send OTP]
    G --> H[Enter OTP]
    H --> I{OTP Valid?}
    I -->|Yes| F
    I -->|No| J{Retries Left?}
    J -->|Yes| G
    J -->|No| K[Lock Account]"#,
            MermaidTheme::dark(),
        ),
    ];

    std::fs::create_dir_all("/tmp/mermaid_showcase").unwrap();

    for (name, src, theme) in &diagrams {
        let diagram = Mermaid::parse(src).unwrap().theme(theme.clone());
        let rendered = diagram.render(800, 600);
        let png = diagram.render_to_png(800, 600).unwrap();
        let path = format!("/tmp/mermaid_showcase/{name}.png");
        std::fs::write(&path, &png).unwrap();
        println!(
            "{name}: {}x{} px, {} bytes, {} draw commands",
            rendered.width,
            rendered.height,
            png.len(),
            rendered.canvas.command_count()
        );
    }
    println!("\nAll diagrams written to /tmp/mermaid_showcase/");
}
