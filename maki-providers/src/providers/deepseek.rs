use std::sync::{Arc, Mutex};

use flume::Sender;
use serde_json::{Value, json};

use crate::model::Model;
use crate::model::{ModelEntry, ModelFamily, ModelPricing, ModelTier};
use crate::provider::{BoxFuture, Provider};
use crate::{
    AgentError, ContentBlock, Message, ProviderEvent, Role, StreamResponse, ThinkingConfig,
};

use super::ResolvedAuth;
use super::openai_compat::{OpenAiCompatConfig, OpenAiCompatProvider, convert_tools};

static CONFIG: OpenAiCompatConfig = OpenAiCompatConfig {
    api_key_env: "DEEPSEEK_API_KEY",
    base_url: "https://api.deepseek.com",
    max_tokens_field: "max_tokens",
    include_stream_usage: false,
    provider_name: "DeepSeek",
};

pub(crate) fn models() -> &'static [ModelEntry] {
    &[
        ModelEntry {
            prefixes: &["deepseek-v4-flash"],
            tier: ModelTier::Medium,
            family: ModelFamily::Gpt,
            default: true,
            pricing: ModelPricing {
                input: 0.14,
                output: 0.28,
                cache_write: 0.00,
                cache_read: 0.0028,
            },
            max_output_tokens: 384_000,
            context_window: 1_000_000,
        },
        ModelEntry {
            prefixes: &["deepseek-v4-pro"],
            tier: ModelTier::Strong,
            family: ModelFamily::Gpt,
            default: true,
            pricing: ModelPricing {
                input: 1.74,
                output: 3.48,
                cache_write: 0.00,
                cache_read: 0.0145,
            },
            max_output_tokens: 384_000,
            context_window: 1_000_000,
        },
    ]
}

pub struct DeepSeek {
    compat: OpenAiCompatProvider,
    auth: Arc<Mutex<ResolvedAuth>>,
    system_prefix: Option<String>,
}

impl DeepSeek {
    pub fn new(timeouts: super::Timeouts) -> Result<Self, AgentError> {
        let api_key = std::env::var(CONFIG.api_key_env).map_err(|_| AgentError::Config {
            message: format!("{} not set", CONFIG.api_key_env),
        })?;
        Ok(Self {
            compat: OpenAiCompatProvider::new(&CONFIG, timeouts),
            auth: Arc::new(Mutex::new(ResolvedAuth::bearer(&api_key))),
            system_prefix: None,
        })
    }

    pub(crate) fn with_auth(auth: Arc<Mutex<ResolvedAuth>>, timeouts: super::Timeouts) -> Self {
        Self {
            compat: OpenAiCompatProvider::new(&CONFIG, timeouts),
            auth,
            system_prefix: None,
        }
    }

    pub(crate) fn with_system_prefix(mut self, prefix: Option<String>) -> Self {
        self.system_prefix = prefix;
        self
    }

    pub fn build_body(
        &self,
        model: &crate::model::Model,
        messages: &[Message],
        system: &str,
        tools: &Value,
    ) -> Value {
        let wire_messages = convert_messages(messages, system);
        let wire_tools = convert_tools(tools);

        let mut body = json!({
            "model": model.id,
            "messages": wire_messages,
            "stream": true,
            CONFIG.max_tokens_field: model.max_output_tokens,
        });
        if CONFIG.include_stream_usage {
            body["stream_options"] = json!({"include_usage": true});
        }
        if wire_tools.as_array().is_some_and(|a| !a.is_empty()) {
            body["tools"] = wire_tools;
        }
        body
    }
}

pub fn convert_messages(messages: &[Message], system: &str) -> Vec<Value> {
    let mut out = vec![json!({"role": "system", "content": system})];
    let mut used_tool = false;

    for msg in messages {
        match msg.role {
            Role::User => {
                let mut tool_results = Vec::new();
                let mut text_parts: Vec<&str> = Vec::new();
                let mut image_parts = Vec::new();

                for block in &msg.content {
                    match block {
                        ContentBlock::Text { text } => text_parts.push(text.as_str()),
                        ContentBlock::Image { source } => {
                            image_parts.push(json!({
                                "type": "image_url",
                                "image_url": { "url": source.to_data_url() }
                            }));
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            tool_results.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_use_id,
                                "content": content,
                            }));
                        }
                        ContentBlock::ToolUse { .. }
                        | ContentBlock::Thinking { .. }
                        | ContentBlock::RedactedThinking { .. } => {}
                    }
                }

                if !image_parts.is_empty() {
                    let mut parts = image_parts;
                    if !text_parts.is_empty() {
                        parts.push(json!({"type": "text", "text": text_parts.join("\n")}));
                    }
                    out.push(json!({"role": "user", "content": parts}));
                } else if !text_parts.is_empty() {
                    out.push(json!({"role": "user", "content": text_parts.join("\n")}));
                }
                out.extend(tool_results);
            }
            Role::Assistant => {
                let mut text = String::new();
                let mut tool_calls = Vec::new();
                let mut reasoning_content = String::new();

                for block in &msg.content {
                    match block {
                        ContentBlock::Text { text: t } => text.push_str(t),
                        ContentBlock::ToolUse { id, name, input } => {
                            used_tool = true;
                            tool_calls.push(json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": input.to_string(),
                                }
                            }));
                        }
                        ContentBlock::Thinking { thinking, .. } => {
                            reasoning_content.push_str(thinking);
                        }
                        ContentBlock::ToolResult { .. }
                        | ContentBlock::Image { .. }
                        | ContentBlock::RedactedThinking { .. } => {}
                    }
                }

                if !text.is_empty() || !tool_calls.is_empty() || !reasoning_content.is_empty() {
                    let mut msg_obj = json!({"role": "assistant"});
                    if !text.is_empty() {
                        msg_obj["content"] = Value::String(text);
                    }
                    if !tool_calls.is_empty() {
                        msg_obj["tool_calls"] = Value::Array(tool_calls);
                    }
                    // for turns that do perform tool calls, the reasoning_content must be fully passed back to the API in all subsequent requests
                    if used_tool && !reasoning_content.is_empty() {
                        msg_obj["reasoning_content"] = Value::String(reasoning_content);
                    }
                    out.push(msg_obj);
                }
            }
        }
    }

    out
}

impl Provider for DeepSeek {
    fn stream_message<'a>(
        &'a self,
        model: &'a Model,
        messages: &'a [Message],
        system: &'a str,
        tools: &'a Value,
        event_tx: &'a Sender<ProviderEvent>,
        _thinking: ThinkingConfig,
        _session_id: Option<&str>,
    ) -> BoxFuture<'a, Result<StreamResponse, AgentError>> {
        Box::pin(async move {
            let auth = self.auth.lock().unwrap().clone();
            let mut buf = String::new();
            let system = super::with_prefix(&self.system_prefix, system, &mut buf);
            let body = self.build_body(model, messages, system, tools);
            self.compat
                .do_stream(model, &[], &body, event_tx, &auth)
                .await
        })
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>, AgentError>> {
        Box::pin(async move {
            let auth = self.auth.lock().unwrap().clone();
            self.compat.do_list_models(&auth).await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Model, ModelFamily, ModelPricing, ModelTier};
    use crate::provider::ProviderKind;
    use crate::{ContentBlock, Message, Role};

    fn create_test_model() -> Model {
        Model {
            id: "deepseek-v4-pro".to_string(),
            provider: ProviderKind::Deepseek,
            dynamic_slug: None,
            tier: ModelTier::Strong,
            family: ModelFamily::Gpt,
            supports_tool_examples_override: None,
            pricing: ModelPricing {
                input: 1.74,
                output: 3.48,
                cache_write: 0.00,
                cache_read: 0.0145,
            },
            max_output_tokens: 384_000,
            context_window: 1_000_000,
        }
    }

    #[test]
    fn test_build_body_basic_conversation() {
        let auth = Arc::new(Mutex::new(ResolvedAuth::bearer("test_key")));
        let provider = DeepSeek::with_auth(auth, crate::providers::Timeouts::default());
        let model = create_test_model();
        let messages = vec![
            Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: "Hello, how are you?".to_string(),
                }],
                display_text: None,
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: "I'm doing well, thank you!".to_string(),
                }],
                display_text: None,
            },
        ];
        let system = "You are a helpful assistant.";
        let tools = json!({});

        let body = provider.build_body(&model, &messages, system, &tools);

        assert_eq!(body["model"], "deepseek-v4-pro");
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], 384_000);

        let messages_out = body["messages"].as_array().unwrap();
        assert_eq!(messages_out.len(), 3); // system + user + assistant
        assert_eq!(messages_out[0]["role"], "system");
        assert_eq!(messages_out[0]["content"], "You are a helpful assistant.");
        assert_eq!(messages_out[1]["role"], "user");
        assert_eq!(messages_out[1]["content"], "Hello, how are you?");
        assert_eq!(messages_out[2]["role"], "assistant");
        assert_eq!(messages_out[2]["content"], "I'm doing well, thank you!");
    }

    #[test]
    fn test_build_body_reasoning_content_included_with_tool_calls() {
        let auth = Arc::new(Mutex::new(ResolvedAuth::bearer("test_key")));
        let provider = DeepSeek::with_auth(auth, crate::providers::Timeouts::default());
        let model = create_test_model();

        let messages = vec![
            Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: "Solve this problem".to_string(),
                }],
                display_text: None,
            },
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Thinking {
                        thinking: "Let me think about this...".to_string(),
                        signature: None,
                    },
                    ContentBlock::ToolUse {
                        id: "call_456".to_string(),
                        name: "calculate".to_string(),
                        input: json!({"expression": "2+2"}),
                    },
                ],
                display_text: None,
            },
        ];
        let system = "You are a helpful assistant.";
        let tools = json!({});

        let body = provider.build_body(&model, &messages, system, &tools);
        let messages_out = body["messages"].as_array().unwrap();
        let assistant_msg = &messages_out[2];

        // After a tool call, reasoning_content SHOULD be included
        assert!(
            assistant_msg.get("reasoning_content").is_some(),
            "reasoning_content should be present when there's a tool call"
        );
        assert_eq!(
            assistant_msg["reasoning_content"],
            "Let me think about this..."
        );
    }

    #[test]
    fn test_build_body_reasoning_content_excluded_without_tool_calls() {
        let auth = Arc::new(Mutex::new(ResolvedAuth::bearer("test_key")));
        let provider = DeepSeek::with_auth(auth, crate::providers::Timeouts::default());
        let model = create_test_model();

        let messages = vec![
            Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: "Tell me a joke".to_string(),
                }],
                display_text: None,
            },
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Thinking {
                        thinking: "Thinking of a funny joke...".to_string(),
                        signature: None,
                    },
                    ContentBlock::Text {
                        text: "Why did the chicken cross the road?".to_string(),
                    },
                ],
                display_text: None,
            },
        ];

        let system = "You are a helpful assistant.";
        let tools = json!({});

        let body = provider.build_body(&model, &messages, system, &tools);
        let messages_out = body["messages"].as_array().unwrap();
        let assistant_msg = &messages_out[2];

        // Without tool calls, reasoning_content should NOT be included
        assert!(
            assistant_msg.get("reasoning_content").is_none(),
            "reasoning_content should NOT be present when there's no tool call"
        );
        assert_eq!(
            assistant_msg["content"],
            "Why did the chicken cross the road?"
        );
    }
}
