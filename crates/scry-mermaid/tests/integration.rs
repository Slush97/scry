use scry_mermaid::prelude::*;

#[test]
fn parse_and_render_flowchart() {
    let src = r#"graph TD
    A[Start] --> B{Is it working?}
    B -->|Yes| C([Deploy])
    B -->|No| D[Debug]
    D --> B
    C --> E((Done))"#;

    let diagram = Mermaid::parse(src).unwrap();
    let rendered = diagram.render(800, 600);

    // Verify we got a non-trivial canvas.
    assert!(rendered.width > 100);
    assert!(rendered.height > 100);

    // Verify the canvas has draw commands (nodes + edges + text).
    assert!(rendered.canvas.command_count() > 10);
}

#[test]
fn render_to_png() {
    let src = "graph LR\n    A[Input] --> B[Process] --> C[Output]";
    let diagram = Mermaid::parse(src).unwrap();
    let png = diagram.render_to_png(800, 600).unwrap();

    // PNG magic bytes.
    assert_eq!(&png[..4], &[0x89, 0x50, 0x4E, 0x47]);
    assert!(png.len() > 1000, "PNG should be non-trivial, got {} bytes", png.len());
}

#[test]
fn theme_customization() {
    let src = "graph TD\n    A --> B";
    let diagram = Mermaid::parse(src)
        .unwrap()
        .theme(MermaidTheme::light());
    let rendered = diagram.render(800, 600);
    assert!(rendered.canvas.command_count() > 0);
}

#[test]
fn all_node_shapes() {
    let src = r#"graph TD
    A[Rectangle] --> B(Rounded)
    B --> C{Diamond}
    C --> D([Stadium])
    D --> E[[Subroutine]]
    E --> F((Circle))"#;

    let diagram = Mermaid::parse(src).unwrap();
    let rendered = diagram.render(800, 600);
    assert!(rendered.canvas.command_count() > 20);
}

#[test]
fn all_edge_styles() {
    let src = r#"graph TD
    A --> B
    B --- C
    C -.-> D
    D ==> E"#;

    let diagram = Mermaid::parse(src).unwrap();
    let rendered = diagram.render(800, 600);
    assert!(rendered.canvas.command_count() > 10);
}

#[test]
fn horizontal_layout() {
    let src = "flowchart LR\n    A --> B --> C";
    let diagram = Mermaid::parse(src).unwrap();
    let rendered = diagram.render(800, 600);

    // In LR layout, width should generally exceed height for a chain.
    assert!(rendered.width > rendered.height / 2);
}

#[test]
fn render_with_back_edges() {
    let src = r#"graph TD
    A[Start] --> B{Check}
    B -->|Yes| C[Done]
    B -->|No| D[Retry]
    D --> A"#;

    let diagram = Mermaid::parse(src).unwrap();
    let rendered = diagram.render(800, 600);

    // Should render without panicking.
    assert!(rendered.canvas.command_count() > 10);
    // TD layout should be portrait-ish, not a flat line.
    assert!(
        rendered.height > rendered.width / 3,
        "Should be taller than a flat line: {}x{}",
        rendered.width,
        rendered.height
    );
}

#[test]
fn render_complex_with_cycles() {
    let src = r#"graph TD
    A[Push] --> B{Tests?}
    B -->|Yes| C[Build]
    B -->|No| D[Notify]
    D --> E[Fix]
    E --> A
    C --> F{Deploy?}
    F -->|Yes| G[Staging]
    G --> H{Smoke?}
    H -->|Yes| I[Prod]
    H -->|No| D
    F -->|No| J[End]"#;

    let diagram = Mermaid::parse(src).unwrap();
    let png = diagram.render_to_png(800, 600).unwrap();
    assert_eq!(&png[..4], &[0x89, 0x50, 0x4E, 0x47]);
    assert!(png.len() > 1000);
}

#[test]
fn render_self_loop() {
    let src = "graph TD\n    A --> A";
    let diagram = Mermaid::parse(src).unwrap();
    let rendered = diagram.render(800, 600);
    assert!(rendered.canvas.command_count() > 0);
}

#[test]
fn render_single_node() {
    let src = "graph TD\n    A[Hello World]";
    let diagram = Mermaid::parse(src).unwrap();
    let rendered = diagram.render(800, 600);
    assert!(rendered.canvas.command_count() > 0);
}

#[test]
fn render_dotted_with_label() {
    let src = "graph TD\n    A -.->|maybe| B";
    let diagram = Mermaid::parse(src).unwrap();
    let rendered = diagram.render(800, 600);
    // Should have: 2 nodes (rect+text each) + 1 edge + 1 arrowhead + label bg + label text = ~8+
    assert!(rendered.canvas.command_count() >= 6);
}

#[test]
fn render_scales_to_fit_bounds() {
    let src = r#"graph TD
    A[Start] --> B --> C --> D --> E --> F --> G --> H --> I --> J[End]"#;
    let diagram = Mermaid::parse(src).unwrap();

    // Render unconstrained (large bounds).
    let big = diagram.render(10000, 10000);

    // Render with tight bounds.
    let small = diagram.render(200, 200);

    // The constrained render should be smaller or equal to the bounds.
    assert!(small.width <= 200, "width {} should be <= 200", small.width);
    assert!(small.height <= 200, "height {} should be <= 200", small.height);

    // And it should be smaller than the unconstrained render.
    assert!(
        small.width < big.width || small.height < big.height,
        "constrained render should be smaller: {}x{} vs {}x{}",
        small.width, small.height, big.width, big.height
    );
}

#[test]
fn render_no_upscale_when_within_bounds() {
    let src = "graph TD\n    A --> B";
    let diagram = Mermaid::parse(src).unwrap();

    let normal = diagram.render(10000, 10000);
    let same = diagram.render(5000, 5000);

    // When bounds are larger than natural size, output should be the same.
    assert_eq!(normal.width, same.width);
    assert_eq!(normal.height, same.height);
}

#[test]
fn unsupported_diagram_type() {
    let result = Mermaid::parse("sequenceDiagram\n    A->>B: hello");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, MermaidError::UnsupportedDiagram(_)));
}
