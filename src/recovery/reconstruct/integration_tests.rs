//! Integration tests for the reconstruction pipeline.
//!
//! These tests construct synthetic IR statement trees, run the full
//! reconstruction pipeline (`apply`), and verify the final rendered output.
//! They test both individual patterns and combinations of patterns working
//! together.

#[cfg(test)]
mod tests {
    use crate::recovery::csharp::{render_stmts, BinOp, CmpKind, Expr, Stmt};
    use crate::recovery::reconstruct::{apply, metrics_summary};

    /// Helper: render a variable expression.
    fn v(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    /// Helper: render a Monitor.Enter call.
    fn monitor_enter(arg: Expr, lock_taken: bool) -> Stmt {
        let mut args = vec![arg];
        if lock_taken {
            args.push(Expr::Int(0));
        }
        Stmt::Eval(Expr::Call {
            recv: Some(Box::new(v("Monitor"))),
            name: "Enter".into(),
            args,
        })
    }

    /// Helper: render a Monitor.Exit call.
    fn monitor_exit(arg: Expr) -> Stmt {
        Stmt::Eval(Expr::Call {
            recv: Some(Box::new(v("Monitor"))),
            name: "Exit".into(),
            args: vec![arg],
        })
    }

    // -----------------------------------------------------------------------
    // Null-coalescing assignment integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn null_coalesce_renders_correctly() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("x")), Box::new(Expr::Null)),
            then_body: vec![Stmt::Assign("x".into(), None, v("default_val"))],
            else_body: vec![],
        }];
        apply(&mut stmts);
        let output = render_stmts(&stmts);
        assert!(
            output.contains("??="),
            "output should contain ??= operator:\n{output}"
        );
        assert!(
            output.contains("x ??= default_val"),
            "output should contain the full null-coalesce expression:\n{output}"
        );
    }

    #[test]
    fn null_coalesce_with_complex_value() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("cache")), Box::new(Expr::Null)),
            then_body: vec![Stmt::Assign(
                "cache".into(),
                None,
                Expr::Call {
                    recv: Some(Box::new(v("factory"))),
                    name: "Create".into(),
                    args: vec![Expr::Int(42)],
                },
            )],
            else_body: vec![],
        }];
        apply(&mut stmts);
        let output = render_stmts(&stmts);
        assert!(
            output.contains("cache ??= factory.Create(42)"),
            "complex value should be preserved:\n{output}"
        );
    }

    // -----------------------------------------------------------------------
    // For-loop reconstruction integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn for_loop_basic_reconstruction() {
        // init; while (cond) { body; incr; }
        let mut stmts = vec![
            Stmt::Assign("i".into(), Some("int".into()), Expr::Int(0)),
            Stmt::While {
                cond: Expr::Cmp(CmpKind::Lt, Box::new(v("i")), Box::new(Expr::Int(10))),
                body: vec![
                    Stmt::Eval(Expr::Call {
                        recv: None,
                        name: "Process".into(),
                        args: vec![v("i")],
                    }),
                    Stmt::Assign(
                        "i".into(),
                        None,
                        Expr::Bin(BinOp::Add, Box::new(v("i")), Box::new(Expr::Int(1))),
                    ),
                ],
            },
        ];
        apply(&mut stmts);
        let output = render_stmts(&stmts);
        assert!(
            output.contains("for"),
            "output should contain a for loop:\n{output}"
        );
    }

    // -----------------------------------------------------------------------
    // Lock reconstruction integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn lock_basic_reconstruction() {
        let mut stmts = vec![
            monitor_enter(v("gate"), true),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "CriticalWork".into(),
                args: vec![],
            }),
            monitor_exit(v("gate")),
        ];
        apply(&mut stmts);
        let output = render_stmts(&stmts);
        assert!(
            output.contains("lock"),
            "output should contain a lock statement:\n{output}"
        );
    }

    // -----------------------------------------------------------------------
    // Array initializer integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn array_init_reconstruction() {
        // newarr int; stelem 0; stelem 1; stelem 2
        let mut stmts = vec![
            Stmt::Assign("arr".into(), Some("int[]".into()), Expr::NewArr("int".into(), vec![Expr::Int(3)])),
            Stmt::Assign(
                "".into(),
                None,
                Expr::Call {
                    recv: None,
                    name: "stelem".into(),
                    args: vec![v("arr"), Expr::Int(0), Expr::Int(10)],
                },
            ),
        ];
        // Note: array_init keys on specific shapes; this test verifies the
        // pipeline doesn't crash on the input.
        apply(&mut stmts);
        let output = render_stmts(&stmts);
        // The pipeline should produce valid output without panicking.
        assert!(!output.is_empty() || stmts.is_empty());
    }

    // -----------------------------------------------------------------------
    // Combined pattern tests
    // -----------------------------------------------------------------------

    #[test]
    fn null_coalesce_inside_for_loop() {
        // A null-coalesce pattern inside a for-loop body.
        let mut stmts = vec![
            Stmt::Assign("i".into(), Some("int".into()), Expr::Int(0)),
            Stmt::While {
                cond: Expr::Cmp(CmpKind::Lt, Box::new(v("i")), Box::new(Expr::Int(5))),
                body: vec![
                    Stmt::If {
                        cond: Expr::Cmp(CmpKind::Eq, Box::new(v("item")), Box::new(Expr::Null)),
                        then_body: vec![Stmt::Assign("item".into(), None, v("fallback"))],
                        else_body: vec![],
                    },
                    Stmt::Assign(
                        "i".into(),
                        None,
                        Expr::Bin(BinOp::Add, Box::new(v("i")), Box::new(Expr::Int(1))),
                    ),
                ],
            },
        ];
        apply(&mut stmts);
        let output = render_stmts(&stmts);
        // Both patterns should be reconstructed.
        assert!(
            output.contains("??="),
            "null-coalesce should be present inside the loop:\n{output}"
        );
    }

    #[test]
    fn lock_inside_if_branch() {
        // A lock pattern inside an if-then branch.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![
                monitor_enter(v("gate"), true),
                Stmt::Eval(Expr::Call {
                    recv: None,
                    name: "Work".into(),
                    args: vec![],
                }),
                monitor_exit(v("gate")),
            ],
            else_body: vec![],
        }];
        apply(&mut stmts);
        let output = render_stmts(&stmts);
        assert!(
            output.contains("lock"),
            "lock should be reconstructed inside the if:\n{output}"
        );
    }

    #[test]
    fn multiple_null_coalesces_in_sequence() {
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Eq, Box::new(v("a")), Box::new(Expr::Null)),
                then_body: vec![Stmt::Assign("a".into(), None, v("default_a"))],
                else_body: vec![],
            },
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Eq, Box::new(v("b")), Box::new(Expr::Null)),
                then_body: vec![Stmt::Assign("b".into(), None, v("default_b"))],
                else_body: vec![],
            },
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Eq, Box::new(v("c")), Box::new(Expr::Null)),
                then_body: vec![Stmt::Assign("c".into(), None, v("default_c"))],
                else_body: vec![],
            },
        ];
        apply(&mut stmts);
        let output = render_stmts(&stmts);
        let coalesce_count = output.matches("??=").count();
        assert_eq!(
            coalesce_count, 3,
            "all three null-coalesces should be reconstructed:\n{output}"
        );
    }

    #[test]
    fn null_coalesce_not_applied_when_else_present() {
        // An if-else with null check should NOT be rewritten to ??=.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("x")), Box::new(Expr::Null)),
            then_body: vec![Stmt::Assign("x".into(), None, v("a"))],
            else_body: vec![Stmt::Assign("x".into(), None, v("b"))],
        }];
        apply(&mut stmts);
        let output = render_stmts(&stmts);
        assert!(
            !output.contains("??="),
            "if-else should not become ??=:\n{output}"
        );
    }

    // -----------------------------------------------------------------------
    // Ctor normalization integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn ctor_normalization_enables_downstream_passes() {
        // A ctor call should be normalized to NewObj, enabling downstream passes.
        let mut stmts = vec![Stmt::Assign(
            "list".into(),
            None,
            Expr::Call {
                recv: None,
                name: "List<int>..ctor".into(),
                args: vec![Expr::Int(0)],
            },
        )];
        apply(&mut stmts);
        let output = render_stmts(&stmts);
        // After normalization, the ctor call becomes NewObj.
        assert!(
            output.contains("new List<int>"),
            "ctor should be normalized to new:\n{output}"
        );
    }

    // -----------------------------------------------------------------------
    // Pipeline robustness tests
    // -----------------------------------------------------------------------

    #[test]
    fn empty_body_does_not_panic() {
        let mut stmts: Vec<Stmt> = vec![];
        apply(&mut stmts);
        // Should not panic.
    }

    #[test]
    fn deeply_nested_patterns() {
        // Patterns nested several levels deep should still be found.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::If {
                    cond: Expr::Bool(true),
                    then_body: vec![Stmt::If {
                        cond: Expr::Cmp(CmpKind::Eq, Box::new(v("x")), Box::new(Expr::Null)),
                        then_body: vec![Stmt::Assign("x".into(), None, v("deep_default"))],
                        else_body: vec![],
                    }],
                    else_body: vec![],
                }],
                else_body: vec![],
            }],
            else_body: vec![],
        }];
        apply(&mut stmts);
        let output = render_stmts(&stmts);
        assert!(
            output.contains("??="),
            "deeply nested null-coalesce should be found:\n{output}"
        );
    }

    #[test]
    fn pattern_inside_try_catch() {
        let mut stmts = vec![Stmt::Try {
            try_body: vec![Stmt::If {
                cond: Expr::Cmp(CmpKind::Eq, Box::new(v("x")), Box::new(Expr::Null)),
                then_body: vec![Stmt::Assign("x".into(), None, v("default"))],
                else_body: vec![],
            }],
            catches: vec![(
                "Exception".into(),
                vec![Stmt::Eval(Expr::Call {
                    recv: None,
                    name: "Handle".into(),
                    args: vec![],
                })],
            )],
            finally_body: None,
        }];
        apply(&mut stmts);
        let output = render_stmts(&stmts);
        assert!(
            output.contains("??="),
            "null-coalesce inside try should be found:\n{output}"
        );
    }

    #[test]
    fn pattern_inside_while_loop() {
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![
                Stmt::If {
                    cond: Expr::Cmp(CmpKind::Eq, Box::new(v("x")), Box::new(Expr::Null)),
                    then_body: vec![Stmt::Assign("x".into(), None, v("default"))],
                    else_body: vec![],
                },
                Stmt::Break,
            ],
        }];
        apply(&mut stmts);
        let output = render_stmts(&stmts);
        assert!(
            output.contains("??="),
            "null-coalesce inside while should be found:\n{output}"
        );
    }

    // -----------------------------------------------------------------------
    // Report/metrics tests
    // -----------------------------------------------------------------------

    #[test]
    fn report_includes_metrics_after_rewrites() {
        use crate::recovery::reconstruct::{metrics_summary, report};

        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("x")), Box::new(Expr::Null)),
            then_body: vec![Stmt::Assign("x".into(), None, v("default"))],
            else_body: vec![],
        }];
        apply(&mut stmts);

        let r = report();
        // The report should mention the null_coalesce pass.
        assert!(
            r.contains("null_coalesce") || r.contains("construct rewrite"),
            "report should mention reconstruction activity:\n{r}"
        );

        let m = metrics_summary();
        assert!(
            m.contains("metrics dashboard"),
            "metrics summary should include dashboard header:\n{m}"
        );
    }

    // -----------------------------------------------------------------------
    // IL2CPP de-weaving integration tests (Phase 7)
    // -----------------------------------------------------------------------

    #[test]
    fn il2cpp_null_guard_is_stripped_from_output() {
        // if (this.m_CachedPtr == 0) { ThrowHelper.ThrowNullReferenceException(this); }
        // doWork();
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Eq,
                    Box::new(Expr::Field(Box::new(Expr::This), "m_CachedPtr".into())),
                    Box::new(Expr::Int(0)),
                ),
                then_body: vec![Stmt::Eval(Expr::Call {
                    recv: Some(Box::new(v("ThrowHelper"))),
                    name: "ThrowNullReferenceException".into(),
                    args: vec![Expr::This],
                })],
                else_body: vec![],
            },
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Field(
                    Box::new(Expr::This),
                    "m_CachedPtr".into(),
                ))),
                name: "doWork".into(),
                args: vec![],
            }),
        ];
        apply(&mut stmts);
        let output = render_stmts(&stmts);
        assert!(
            !output.contains("ThrowNullReferenceException"),
            "null guard should be stripped from output:\n{output}"
        );
        assert!(
            output.contains("doWork()"),
            "real work should be preserved:\n{output}"
        );
    }

    #[test]
    fn il2cpp_class_init_guard_is_stripped_from_output() {
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Eq,
                    Box::new(Expr::RawDeref(Box::new(Expr::Int(0x25b22c0)), 0, 1)),
                    Box::new(Expr::Int(0)),
                ),
                then_body: vec![Stmt::Eval(Expr::Call {
                    recv: None,
                    name: "IL2CPP_ClassInit".into(),
                    args: vec![Expr::RawDeref(Box::new(Expr::Int(0x2479a50)), 0, 8)],
                })],
                else_body: vec![],
            },
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "realWork".into(),
                args: vec![],
            }),
        ];
        apply(&mut stmts);
        let output = render_stmts(&stmts);
        assert!(
            !output.contains("IL2CPP_ClassInit"),
            "class init guard should be stripped:\n{output}"
        );
        assert!(
            output.contains("realWork()"),
            "real work should be preserved:\n{output}"
        );
    }

    #[test]
    fn rpc_ternary_collapses_to_this_in_output() {
        // (v58 == 0 ? this : v58).CallRpcRemove();
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: Some(Box::new(Expr::Cond(
                Box::new(Expr::Cmp(CmpKind::Eq, Box::new(v("v58")), Box::new(Expr::Int(0)))),
                Box::new(Expr::This),
                Box::new(v("v58")),
            ))),
            name: "CallRpcRemove".into(),
            args: vec![],
        })];
        apply(&mut stmts);
        let output = render_stmts(&stmts);
        assert!(
            !output.contains("? this :"),
            "ternary should be collapsed:\n{output}"
        );
        assert!(
            output.contains("this.CallRpcRemove()"),
            "should render as this.CallRpcRemove():\n{output}"
        );
    }

    #[test]
    fn temp_var_flattened_into_condition_in_output() {
        // v33 = DiscreteTime.op_Inequality(v27, v30);
        // if (v33 != 0) { return 0; }
        let mut stmts = vec![
            Stmt::Assign(
                "v33".into(),
                None,
                Expr::Call {
                    recv: None,
                    name: "DiscreteTime.op_Inequality".into(),
                    args: vec![v("v27"), v("v30")],
                },
            ),
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ne, Box::new(v("v33")), Box::new(Expr::Int(0))),
                then_body: vec![Stmt::Return(Some(Expr::Int(0)))],
                else_body: vec![],
            },
        ];
        apply(&mut stmts);
        let output = render_stmts(&stmts);
        assert!(
            !output.contains("v33 ="),
            "temp assignment should be inlined:\n{output}"
        );
        // The inlined call is ALSO folded to its operator: `op_Inequality` IS
        // C#'s `!=` (that is the required declaration name), so the condition
        // reads `v27 != v30`. The assertion still proves what this test is for
        // — the temp was inlined into the condition rather than left as a
        // statement — and now proves the operator fold too.
        assert!(
            output.contains("v27 != v30"),
            "call should be inlined into the condition and folded to `!=`:\n{output}"
        );
    }

    #[test]
    fn il2cpp_cleanup_appears_in_metrics_summary() {
        // Run a pipeline that triggers il2cpp_cleanup.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("p")), Box::new(Expr::Null)),
            then_body: vec![Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("ThrowHelper"))),
                name: "ThrowNullReferenceException".into(),
                args: vec![v("p")],
            })],
            else_body: vec![],
        }];
        apply(&mut stmts);
        let m = metrics_summary();
        assert!(
            m.contains("IL2CPP de-weaving") || m.contains("il2cpp_cleanup"),
            "metrics should mention il2cpp_cleanup:\n{m}"
        );
    }
}
