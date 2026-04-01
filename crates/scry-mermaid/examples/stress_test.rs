use scry_mermaid::prelude::*;

fn main() {
    let diagrams: Vec<(&str, &str, MermaidTheme)> = vec![
        // 1. Compiler pipeline — deep linear chain
        (
            "compiler_pipeline",
            r#"graph TD
    A[Source Code] --> B[Lexer]
    B --> C[Token Stream]
    C --> D[Parser]
    D --> E[AST]
    E --> F{Type Check}
    F -->|Pass| G[HIR]
    F -->|Fail| H[Error Report]
    H --> A
    G --> I[MIR]
    I --> J[Borrow Check]
    J -->|Pass| K[Optimized MIR]
    J -->|Fail| H
    K --> L[LLVM IR]
    L --> M[Machine Code]
    M --> N([Binary])"#,
            MermaidTheme::dark(),
        ),

        // 2. Kubernetes deployment — wide branching
        (
            "k8s_deploy",
            r#"graph TD
    A[Helm Chart] --> B{Environment}
    B -->|Dev| C[dev-cluster]
    B -->|Staging| D[staging-cluster]
    B -->|Prod| E[prod-cluster]
    C --> F[Ingress]
    D --> F
    E --> F
    F --> G[Service]
    G --> H[Pod 1]
    G --> I[Pod 2]
    G --> J[Pod 3]
    H --> K[(PostgreSQL)]
    I --> K
    J --> K
    K --> L[(Redis Cache)]"#,
            MermaidTheme::dark(),
        ),

        // 3. ML training loop — multiple cycles
        (
            "ml_training",
            r#"graph TD
    A[Load Dataset] --> B[Split Train/Val]
    B --> C[Initialize Model]
    C --> D[Forward Pass]
    D --> E[Compute Loss]
    E --> F[Backward Pass]
    F --> G[Update Weights]
    G --> H{Epoch Done?}
    H -->|No| D
    H -->|Yes| I{Val Loss Improved?}
    I -->|Yes| J[Save Checkpoint]
    J --> K{Max Epochs?}
    I -->|No| L[Early Stop Counter]
    L --> M{Patience Exceeded?}
    M -->|No| K
    M -->|Yes| N([Best Model])
    K -->|No| D
    K -->|Yes| N"#,
            MermaidTheme::dark(),
        ),

        // 4. HTTP request lifecycle — LR wide
        (
            "http_lifecycle",
            r#"flowchart LR
    A((Client)) --> B[DNS Resolve]
    B --> C[TCP Handshake]
    C --> D[TLS Negotiate]
    D --> E[Send Request]
    E --> F{Load Balancer}
    F --> G[Server 1]
    F --> H[Server 2]
    G --> I{Cache Hit?}
    H --> I
    I -->|Yes| J[Return Cached]
    I -->|No| K[(Database)]
    K --> L[Serialize]
    J --> M[Response]
    L --> M
    M --> N((Client))"#,
            MermaidTheme::dark(),
        ),

        // 5. Event sourcing — complex with many shapes
        (
            "event_sourcing",
            r#"graph TD
    A((User Action)) --> B[Command Handler]
    B --> C{Validate}
    C -->|Invalid| D[Rejection]
    C -->|Valid| E[Domain Event]
    E --> F[(Event Store)]
    F --> G[[Projector]]
    G --> H[(Read Model)]
    H --> I([Query API])
    F --> J[[Saga Handler]]
    J --> K{Side Effect?}
    K -->|Yes| L[External Service]
    L --> M[Compensation Event]
    M --> F
    K -->|No| N[Complete]
    I --> O((Response))"#,
            MermaidTheme::dark(),
        ),

        // 6. Git merge strategy — decision heavy
        (
            "git_merge",
            r#"graph TD
    A[Incoming PR] --> B{Conflicts?}
    B -->|No| C{CI Green?}
    B -->|Yes| D[Resolve Conflicts]
    D --> C
    C -->|No| E[Fix Tests]
    E --> C
    C -->|Yes| F{Review Approved?}
    F -->|No| G[Request Changes]
    G --> H[Author Revises]
    H --> C
    F -->|Yes| I{Squash?}
    I -->|Yes| J[Squash Merge]
    I -->|No| K{Rebase?}
    K -->|Yes| L[Rebase Merge]
    K -->|No| M[Merge Commit]
    J --> N([Main Branch])
    L --> N
    M --> N"#,
            MermaidTheme::light(),
        ),

        // 7. Database replication — symmetric topology
        (
            "db_replication",
            r#"flowchart LR
    A[(Primary DB)] --> B[WAL Stream]
    B --> C[(Replica 1)]
    B --> D[(Replica 2)]
    B --> E[(Replica 3)]
    C --> F{Sync?}
    D --> F
    E --> F
    F -->|Lag| G[Alert]
    F -->|OK| H[Health Check]
    H --> I([Load Balancer])
    I --> C
    I --> D
    I --> E"#,
            MermaidTheme::dark(),
        ),

        // 8. Incident response — operational flow
        (
            "incident_response",
            r#"graph TD
    A((Alert Fires)) --> B{Severity}
    B -->|P1| C[Page On-Call]
    B -->|P2| D[Slack Channel]
    B -->|P3| E[Ticket Queue]
    C --> F[Acknowledge]
    F --> G[Open War Room]
    G --> H{Root Cause Found?}
    H -->|No| I[Escalate]
    I --> G
    H -->|Yes| J[Apply Fix]
    J --> K{Metrics Recovered?}
    K -->|No| J
    K -->|Yes| L[Write Postmortem]
    L --> M([Close Incident])
    D --> F
    E --> N[Next Sprint]"#,
            MermaidTheme::dark(),
        ),

        // 9. State machine — auth session
        (
            "auth_state_machine",
            r#"graph TD
    A((Unauthenticated)) --> B{Login Attempt}
    B -->|Invalid| C[Failed]
    C --> A
    B -->|Valid| D{MFA Required?}
    D -->|No| E([Authenticated])
    D -->|Yes| F[MFA Pending]
    F --> G{MFA Valid?}
    G -->|Yes| E
    G -->|No| H{Attempts Left?}
    H -->|Yes| F
    H -->|No| I[Locked]
    I --> J[Admin Unlock]
    J --> A
    E --> K{Session Timeout?}
    K -->|Yes| A
    K -->|No| E"#,
            MermaidTheme::dark(),
        ),

        // 10. Data warehouse ETL — wide LR pipeline
        (
            "etl_pipeline",
            r#"flowchart LR
    A[(MySQL)] --> D[[Extract]]
    B[(MongoDB)] --> D
    C[(S3 Bucket)] --> D
    D --> E[Clean]
    E --> F[Deduplicate]
    F --> G{Schema Valid?}
    G -->|Yes| H[Transform]
    G -->|No| I[Dead Letter]
    H --> J[Enrich]
    J --> K[[Load]]
    K --> L[(Snowflake)]
    K --> M[(BigQuery)]
    L --> N([dbt Models])
    M --> N
    N --> O([Dashboard])"#,
            MermaidTheme::dark(),
        ),

        // 11. Minimal — single edge
        (
            "minimal",
            "graph TD\n    A --> B",
            MermaidTheme::dark(),
        ),

        // 12. Minimal — single node
        (
            "single_node",
            "graph TD\n    A[Hello World]",
            MermaidTheme::dark(),
        ),

        // 13. Diamond only — pure decision tree
        (
            "decision_tree",
            r#"graph TD
    A{Age >= 18?}
    A -->|Yes| B{Income >= 50k?}
    A -->|No| C([Reject])
    B -->|Yes| D{Credit Score >= 700?}
    B -->|No| E([Review])
    D -->|Yes| F([Approve])
    D -->|No| E"#,
            MermaidTheme::light(),
        ),

        // 14. Flat wide — stress test horizontal spacing
        (
            "wide_flat",
            r#"graph TD
    A[Router] --> B[Service A]
    A --> C[Service B]
    A --> D[Service C]
    A --> E[Service D]
    A --> F[Service E]
    B --> G([Response])
    C --> G
    D --> G
    E --> G
    F --> G"#,
            MermaidTheme::dark(),
        ),

        // 15. Deep chain — stress test vertical depth
        (
            "deep_chain",
            r#"graph TD
    A[Step 1] --> B[Step 2]
    B --> C[Step 3]
    C --> D[Step 4]
    D --> E[Step 5]
    E --> F[Step 6]
    F --> G[Step 7]
    G --> H[Step 8]
    H --> I[Step 9]
    I --> J[Step 10]
    J --> K([Done])"#,
            MermaidTheme::dark(),
        ),
    ];

    std::fs::create_dir_all("/tmp/mermaid_stress").unwrap();

    for (name, src, theme) in &diagrams {
        let diagram = Mermaid::parse(src).unwrap().theme(theme.clone());
        let rendered = diagram.render(800, 600);
        let png = diagram.render_to_png(800, 600).unwrap();
        let path = format!("/tmp/mermaid_stress/{name}.png");
        std::fs::write(&path, &png).unwrap();
        println!(
            "{name:25} {w:>5}x{h:<5} {kb:>6.1}kb  {cmds:>3} cmds",
            w = rendered.width,
            h = rendered.height,
            kb = png.len() as f64 / 1024.0,
            cmds = rendered.canvas.command_count(),
        );
    }
    println!("\nAll {} diagrams written to /tmp/mermaid_stress/", diagrams.len());
}
