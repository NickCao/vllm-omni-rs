//! HF chat-template rendering via minijinja, fed by a template string
//! fetched once at startup (see `introspect::extract_chat_template`).
//! Zero Python per request: vllm-omni itself delegates all prompt
//! templating to the tokenizer's own Jinja template, and there's no
//! chat-template support in the vendored `vllm-tokenizer` crate, so this
//! renders it ourselves.

use anyhow::{Context, Result};
use minijinja::{Environment, Value, context};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct ChatTemplateRenderer {
    env: Environment<'static>,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplateRenderer {
    pub fn new(
        template: String,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Result<Self> {
        let mut env = Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        // pycompat's string-method shims (.strip(), .split(), .startswith(),
        // etc.) aren't wired in by add_to_environment -- HF chat templates
        // lean on them heavily, so this must be set explicitly.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // HF templates call raise_exception(...) for validation errors;
        // minijinja has no built-in equivalent.
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<(), minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );
        // One process-lifetime template, so a leaked 'static str sidesteps
        // fighting minijinja's Environment<'source> lifetime for an owned
        // String we'd otherwise need to keep alive alongside `env`.
        let template: &'static str = Box::leak(template.into_boxed_str());
        env.add_template("chat", template)
            .context("failed to parse chat_template")?;
        Ok(Self {
            env,
            bos_token: bos_token.unwrap_or_default(),
            eos_token: eos_token.unwrap_or_default(),
        })
    }

    pub fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String> {
        let tmpl = self
            .env
            .get_template("chat")
            .context("chat template not loaded")?;
        tmpl.render(context! {
            messages => messages,
            add_generation_prompt => add_generation_prompt,
            bos_token => self.bos_token,
            eos_token => self.eos_token,
            tools => Value::UNDEFINED,
        })
        .context("chat template render failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A stub in the shape of Qwen's real template (im_start/im_end markers,
    // Python-style reversed iteration, string methods) to validate the
    // context wiring and pycompat setup independent of any real model's
    // template complexity.
    const STUB_TEMPLATE: &str = "\
{%- for message in messages %}\
{{- '<|im_start|>' + message.role + '\n' + message.content.strip() + '<|im_end|>\n' }}\
{%- endfor %}\
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}";

    #[test]
    fn renders_basic_conversation() {
        let renderer = ChatTemplateRenderer::new(
            STUB_TEMPLATE.to_string(),
            None,
            Some("<|im_end|>".to_string()),
        )
        .unwrap();
        let messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "  hi  ".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "hello".into(),
            },
        ];
        let rendered = renderer.render(&messages, true).unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn reversed_iteration_and_length_work() {
        let renderer = ChatTemplateRenderer::new(
            "{%- set ns = namespace(last=messages|length - 1) -%}\
             {%- for m in messages[::-1] -%}{{- m.role }}{%- if not loop.last %},{% endif -%}{%- endfor -%}\
             |{{ ns.last }}"
                .to_string(),
            None,
            None,
        )
        .unwrap();
        let messages = vec![
            ChatMessage {
                role: "a".into(),
                content: String::new(),
            },
            ChatMessage {
                role: "b".into(),
                content: String::new(),
            },
            ChatMessage {
                role: "c".into(),
                content: String::new(),
            },
        ];
        let rendered = renderer.render(&messages, false).unwrap();
        assert_eq!(rendered, "c,b,a|2");
    }
}
