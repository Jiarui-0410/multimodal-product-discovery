package com.productdiscovery.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Component;
import java.util.List;
import java.util.Map;

@Component
public class JsonVectors {
    private final ObjectMapper mapper;
    public JsonVectors(ObjectMapper mapper) { this.mapper = mapper; }

    public String write(Object value) {
        try { return mapper.writeValueAsString(value); }
        catch (JsonProcessingException e) { throw new IllegalStateException("Cannot encode profile data", e); }
    }
    public List<Double> readVector(String json) {
        if (json == null || json.isBlank()) return List.of();
        try { return mapper.readValue(json, new TypeReference<>() {}); }
        catch (JsonProcessingException e) { throw new IllegalStateException("Cannot decode embedding", e); }
    }
    public Map<String, Double> readMap(String json) {
        if (json == null || json.isBlank()) return Map.of();
        try { return mapper.readValue(json, new TypeReference<>() {}); }
        catch (JsonProcessingException e) { throw new IllegalStateException("Cannot decode preferences", e); }
    }
}
